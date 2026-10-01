// Two-view pose diagnostic: run the VO's own two-view geometry on one frame
// pair from a tracks CSV and print every intermediate quantity.
//
//   slam-native-two-view --tracks T.csv --a 100 --b 130 [--hfov 60 | --intrinsics FX FY CX CY]
//                        [--k1 K1] [--points out.csv] [--sweep-hfov]
// With --k1 the tracks are undistorted first (see undistort_point), and all
// pixel coordinates reported or written are undistorted ones.
//
// Steps reported (same functions and thresholds as VisualOdometry::initialize):
//   1. correspondences: tracks observed in both frames, pixel displacement
//   2. pure-rotation fit: best R explaining b = K R K^-1 a (no translation);
//      if its residual is ~ tracking noise, the pair has no usable baseline
//   3. essential matrix RANSAC and homography RANSAC (H/E inlier ratio)
//   4. the four (R, t) decompositions with their cheirality counts
//   5. chosen motion: rotation angle/axis, translation direction, the
//      focus of expansion in pixels, parallax and depth distributions
// --sweep-hfov repeats 3-5 for a range of field-of-view guesses: a wrong focal
// length makes the pair less consistent with any essential matrix.
#include "slam_native/track_io.hpp"
#include "vo_geometry.hpp"

#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

using namespace slam_native;
using namespace slam_native::vo;

namespace {
constexpr double kDegree = 3.14159265358979323846 / 180.0;

double percentile(std::vector<double> v, double q) {
  if (v.empty()) return NAN;
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, static_cast<std::size_t>(q * (v.size() - 1) + 0.5))];
}
std::string quantiles(const std::vector<double>& v) {
  char buffer[160];
  std::snprintf(buffer, sizeof buffer, "p10 %.3g  p50 %.3g  p90 %.3g  (n=%zu)", percentile(v, 0.1),
                percentile(v, 0.5), percentile(v, 0.9), v.size());
  return buffer;
}

struct Pair {
  std::vector<std::uint64_t> ids;
  std::vector<Vec2> ua, ub;  // pixels
};

struct Analysis {
  double hfov{};
  int common{}, e_inliers{}, h_inliers{}, best_count{}, second_count{};
  std::vector<int> candidate_counts;
  SE3 motion;  // camera A = identity, camera B = motion (|t| = 1)
  double rotation_only_residual_px{};  // median, over E inliers
  double rotation_only_angle_deg{};
  std::vector<char> e_inlier, h_inlier, good;
  std::vector<Vec3> X;
  std::vector<double> parallax_deg, reprojection_a, reprojection_b;
};

// Wahba / Kabsch: rotation R minimising sum |b_i - R a_i|^2 over unit bearings.
Mat3 best_rotation(const std::vector<Vec3>& a, const std::vector<Vec3>& b) {
  Mat3 M = Mat3::Zero();
  for (std::size_t i = 0; i < a.size(); ++i) M += b[i] * a[i].transpose();
  Eigen::JacobiSVD<Mat3> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Mat3 D = Mat3::Identity();
  D(2, 2) = (svd.matrixU() * svd.matrixV().transpose()).determinant() < 0 ? -1 : 1;
  return svd.matrixU() * D * svd.matrixV().transpose();
}

Analysis analyze(const Pair& pair, const Intrinsics& K, double hfov, const VisualOdometryConfig& config) {
  Analysis r;
  r.hfov = hfov;
  const int n = static_cast<int>(pair.ids.size());
  r.common = n;
  std::vector<Vec2> a(n), b(n);
  for (int i = 0; i < n; ++i) {
    a[i] = K.unproject(pair.ua[i]).head<2>();
    b[i] = K.unproject(pair.ub[i]).head<2>();
  }
  const double threshold = config.ransac_threshold_px / K.fx;
  const auto E = ransac_essential(a, b, threshold, config.ransac_iterations, 12345);
  const auto H = ransac_homography(a, b, threshold, config.ransac_iterations, 12346);
  r.e_inliers = E.inlier_count;
  r.h_inliers = H.inlier_count;
  r.e_inlier = E.inliers;
  r.h_inlier = H.inliers;

  // Pure rotation, iteratively re-fitted on its own inliers.
  {
    std::vector<char> use(E.inliers.begin(), E.inliers.end());
    Mat3 R = Mat3::Identity();
    std::vector<double> residual(n);
    for (int pass = 0; pass < 3; ++pass) {
      std::vector<Vec3> ba, bb;
      for (int i = 0; i < n; ++i)
        if (use[i]) {
          ba.push_back(K.unproject(pair.ua[i]).normalized());
          bb.push_back(K.unproject(pair.ub[i]).normalized());
        }
      if (ba.size() < 3) break;
      R = best_rotation(ba, bb);
      for (int i = 0; i < n; ++i) {
        const Vec3 p = R * K.unproject(pair.ua[i]);
        residual[i] = p.z() > 0 ? (K.project(p) - pair.ub[i]).norm() : 1e9;
      }
      const double cut = std::max(3.0 * percentile(residual, 0.5), 2.0);
      for (int i = 0; i < n; ++i) use[i] = E.inliers[i] && residual[i] < cut;
    }
    std::vector<double> inlier_residual;
    for (int i = 0; i < n; ++i) if (E.inliers[i]) inlier_residual.push_back(residual[i]);
    r.rotation_only_residual_px = percentile(inlier_residual, 0.5);
    r.rotation_only_angle_deg = log_so3(R).norm() / kDegree;
  }

  const SE3 first;
  const auto candidates = decompose_essential(E.model);
  int best = -1;
  std::vector<std::vector<char>> good_sets;
  std::vector<std::vector<Vec3>> point_sets;
  const double gate = 2 * config.ransac_threshold_px;
  const auto reprojects = [&](const SE3& T, const Vec3& X, const Vec2& u) {
    const Vec3 xc = T * X;
    return xc.z() > 1e-6 && (K.project(xc) - u).norm() <= gate;
  };
  for (std::size_t c = 0; c < candidates.size(); ++c) {
    std::vector<char> good(n, 0);
    std::vector<Vec3> X(n, Vec3::Zero());
    int count = 0;
    for (int i = 0; i < n; ++i) {
      if (!E.inliers[i] || !triangulate(first, a[i], candidates[c], b[i], X[i])) continue;
      good[i] = reprojects(first, X[i], pair.ua[i]) && reprojects(candidates[c], X[i], pair.ub[i]);
      count += good[i];
    }
    r.candidate_counts.push_back(count);
    if (count > r.best_count) {
      r.second_count = r.best_count;
      r.best_count = count;
      best = static_cast<int>(c);
    } else {
      r.second_count = std::max(r.second_count, count);
    }
    good_sets.push_back(std::move(good));
    point_sets.push_back(std::move(X));
  }
  if (best < 0) return r;
  r.motion = candidates[best];
  r.good = good_sets[best];
  r.X = point_sets[best];
  r.parallax_deg.assign(n, NAN);
  r.reprojection_a.assign(n, NAN);
  r.reprojection_b.assign(n, NAN);
  for (int i = 0; i < n; ++i) {
    if (!E.inliers[i]) continue;
    r.parallax_deg[i] = parallax(first, r.motion, r.X[i]) / kDegree;
    const Vec3 xa = r.X[i], xb = r.motion * r.X[i];
    if (xa.z() > 1e-9) r.reprojection_a[i] = (K.project(xa) - pair.ua[i]).norm();
    if (xb.z() > 1e-9) r.reprojection_b[i] = (K.project(xb) - pair.ub[i]).norm();
  }
  return r;
}

void report(const Analysis& r, const Intrinsics& K, const VisualOdometryConfig& config) {
  std::printf("\n== hfov %.1f deg  (fx %.1f px)\n", r.hfov, K.fx);
  std::printf("pure rotation fit: %.2f deg, median residual %.2f px over E inliers\n", r.rotation_only_angle_deg,
              r.rotation_only_residual_px);
  std::printf("essential: %d / %d inliers (%.0f%%)   homography: %d (H/E %.2f, VO waits above %.2f)\n",
              r.e_inliers, r.common, 100.0 * r.e_inliers / std::max(r.common, 1), r.h_inliers,
              double(r.h_inliers) / std::max(r.e_inliers, 1), config.homography_ratio);
  std::printf("decompositions (points in front of both cameras):");
  for (int c : r.candidate_counts) std::printf(" %d", c);
  std::printf("   best/second %d/%d (VO rejects second > 0.7 best)\n", r.best_count, r.second_count);
  if (r.good.empty()) return;
  const Vec3 w = log_so3(r.motion.R);
  const Vec3 axis = w.norm() > 1e-12 ? Vec3(w / w.norm()) : Vec3(0, 0, 0);
  // Camera B centre in camera A's frame is the direction of travel.
  const Vec3 travel = r.motion.center().normalized();
  std::printf("rotation A->B: %.2f deg about (%.2f %.2f %.2f)  [x right, y down, z forward]\n", w.norm() / kDegree,
              axis.x(), axis.y(), axis.z());
  std::printf("travel direction in camera A: (%.3f %.3f %.3f)  angle from optical axis %.1f deg\n", travel.x(),
              travel.y(), travel.z(), std::acos(std::clamp(std::abs(travel.z()), 0.0, 1.0)) / kDegree);
  if (std::abs(travel.z()) > 1e-6) {
    const Vec2 foe = K.project(travel.z() > 0 ? travel : Vec3(-travel));
    std::printf("focus of %s in image A: (%.0f, %.0f) px\n", travel.z() > 0 ? "expansion" : "contraction", foe.x(),
                foe.y());
  }
  std::vector<double> par, par_all, depth, ra, rb;
  int conditioned = 0;
  for (std::size_t i = 0; i < r.good.size(); ++i) {
    if (!r.e_inlier[i]) continue;
    par_all.push_back(r.parallax_deg[i]);
    if (!r.good[i]) continue;
    par.push_back(r.parallax_deg[i]);
    depth.push_back(r.X[i].z());
    ra.push_back(r.reprojection_a[i]);
    rb.push_back(r.reprojection_b[i]);
    conditioned += r.parallax_deg[i] >= config.min_triangulation_parallax_degrees;
  }
  std::printf("parallax deg (good points):   %s\n", quantiles(par).c_str());
  std::printf("well-conditioned (>= %.2f deg): %d  (VO needs %d, median >= %.2f)\n",
              config.min_triangulation_parallax_degrees, conditioned, config.min_init_points,
              config.init_min_parallax_degrees);
  std::printf("depth (baseline = 1):          %s\n", quantiles(depth).c_str());
  std::printf("reprojection px  A: %s\n                 B: %s\n", quantiles(ra).c_str(), quantiles(rb).c_str());
}
}  // namespace

int main(int argc, char** argv) {
  std::string tracks, points_out;
  long long frame_a = -1, frame_b = -1;
  bool sweep = false;
  VisualOdometryConfig config;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string option = argv[i];
      const auto value = [&]() -> std::string {
        if (++i >= argc) throw std::invalid_argument("Missing value for " + option);
        return argv[i];
      };
      if (option == "--tracks") tracks = value();
      else if (option == "--a") frame_a = std::stoll(value());
      else if (option == "--b") frame_b = std::stoll(value());
      else if (option == "--points") points_out = value();
      else if (option == "--sweep-hfov") sweep = true;
      else if (option == "--hfov") config.horizontal_fov_degrees = std::stod(value());
      else if (option == "--k1") config.distortion_k1 = std::stod(value());
      else if (option == "--intrinsics") {
        CameraIntrinsics k;
        k.fx = std::stod(value()); k.fy = std::stod(value()); k.cx = std::stod(value()); k.cy = std::stod(value());
        config.intrinsics = k;
      } else {
        std::cerr << "usage: slam-native-two-view --tracks TRACKS.csv --a FRAME --b FRAME\n"
                     "       [--hfov DEG | --intrinsics FX FY CX CY] [--k1 K1] [--points OUT.csv] [--sweep-hfov]\n";
        return option == "--help" ? 0 : 2;
      }
    }
    if (tracks.empty() || frame_a < 0 || frame_b < 0) throw std::invalid_argument("--tracks, --a and --b are required");
    const auto frames = read_tracks_csv(tracks);
    const TrackedFrame* fa = nullptr;
    const TrackedFrame* fb = nullptr;
    for (const auto& f : frames) {
      if (static_cast<long long>(f.frame_index) == frame_a) fa = &f;
      if (static_cast<long long>(f.frame_index) == frame_b) fb = &f;
    }
    if (!fa || !fb) throw std::invalid_argument("frame not found in tracks");
    const TrackedFrame undistorted_a = undistorted(*fa, config.distortion_k1);
    const TrackedFrame undistorted_b = undistorted(*fb, config.distortion_k1);
    fa = &undistorted_a;
    fb = &undistorted_b;

    Pair pair;
    std::unordered_map<std::uint64_t, Vec2> in_b;
    for (const auto& o : fb->observations) in_b.emplace(o.track_id, Vec2(o.x, o.y));
    std::vector<double> flow;
    for (const auto& o : fa->observations) {
      auto it = in_b.find(o.track_id);
      if (it == in_b.end()) continue;
      pair.ids.push_back(o.track_id);
      pair.ua.emplace_back(o.x, o.y);
      pair.ub.push_back(it->second);
      flow.push_back((it->second - pair.ua.back()).norm());
    }
    std::printf("frames %lld -> %lld (%dx%d): %zu / %zu tracks in common\n", frame_a, frame_b, fa->width,
                fa->height, pair.ids.size(), fa->observations.size());
    std::printf("pixel displacement: %s\n", quantiles(flow).c_str());
    if (pair.ids.size() < 8) throw std::runtime_error("too few common tracks");

    const auto intrinsics_for = [&](double hfov) {
      const auto k = config.intrinsics ? *config.intrinsics :
          CameraIntrinsics::from_horizontal_fov(fa->width, fa->height, hfov);
      return Intrinsics{k.fx, k.fy, k.cx, k.cy};
    };
    const Intrinsics K = intrinsics_for(config.horizontal_fov_degrees);
    const double hfov = 2 * std::atan(0.5 * fa->width / K.fx) / kDegree;
    const Analysis main = analyze(pair, K, hfov, config);
    report(main, K, config);

    if (sweep) {
      std::printf("\nhfov  E-inl  H/E   rot-only-px  rot(E)  travel-vs-axis  median-parallax\n");
      for (double h = 40; h <= 150; h += 10) {
        const Intrinsics Kh = intrinsics_for(h);
        const Analysis s = analyze(pair, Kh, h, config);
        std::vector<double> par;
        for (std::size_t i = 0; i < s.good.size(); ++i) if (s.good[i]) par.push_back(s.parallax_deg[i]);
        const Vec3 travel = s.good.empty() ? Vec3(0, 0, 1) : Vec3(s.motion.center().normalized());
        std::printf("%4.0f  %5d  %.2f  %11.2f  %6.2f  %14.1f  %15.3f\n", h, s.e_inliers,
                    double(s.h_inliers) / std::max(s.e_inliers, 1), s.rotation_only_residual_px,
                    log_so3(s.motion.R).norm() / kDegree,
                    std::acos(std::clamp(std::abs(travel.z()), 0.0, 1.0)) / kDegree, percentile(par, 0.5));
      }
    }

    if (!points_out.empty()) {
      std::ofstream out(points_out);
      const Vec3 cb = main.motion.center();
      out << "# frames " << frame_a << ' ' << frame_b << " size " << fa->width << ' ' << fa->height << " K " << K.fx
          << ' ' << K.fy << ' ' << K.cx << ' ' << K.cy << " cam_b_center " << cb.x() << ' ' << cb.y() << ' '
          << cb.z() << '\n';
      out << "track_id,xa,ya,xb,yb,e_inlier,h_inlier,good,X,Y,Z,parallax_deg,reproj_a,reproj_b\n";
      for (std::size_t i = 0; i < pair.ids.size(); ++i) {
        const bool have = !main.good.empty();
        const Vec3 X = have ? main.X[i] : Vec3(NAN, NAN, NAN);
        out << pair.ids[i] << ',' << pair.ua[i].x() << ',' << pair.ua[i].y() << ',' << pair.ub[i].x() << ','
            << pair.ub[i].y() << ',' << int(main.e_inlier[i]) << ',' << int(main.h_inlier[i]) << ','
            << int(have && main.good[i]) << ',' << X.x() << ',' << X.y() << ',' << X.z() << ','
            << (have ? main.parallax_deg[i] : NAN) << ',' << (have ? main.reprojection_a[i] : NAN) << ','
            << (have ? main.reprojection_b[i] : NAN) << '\n';
      }
      std::printf("\npoints: %s\n", points_out.c_str());
    }
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
