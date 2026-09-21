#include "slam_native/video_decoder.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
}

#include <cerrno>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

namespace slam_native {
namespace {

std::string ffmpeg_error(int code) {
  char buffer[AV_ERROR_MAX_STRING_SIZE]{};
  av_strerror(code, buffer, sizeof(buffer));
  return buffer;
}

void require(int result, const char* operation) {
  if (result < 0) {
    throw std::runtime_error(std::string(operation) + ": " + ffmpeg_error(result));
  }
}

class FfmpegCudaDecoder final : public VideoDecoder {
 public:
  ~FfmpegCudaDecoder() override { close(); }

  void open(const std::filesystem::path& path) override {
    close();
    require(avformat_open_input(&format_, path.c_str(), nullptr, nullptr), "open video");
    require(avformat_find_stream_info(format_, nullptr), "read stream info");
    const int stream = av_find_best_stream(format_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    require(stream, "find video stream");
    stream_index_ = stream;
    stream_ = format_->streams[stream_index_];

    const AVCodec* codec = avcodec_find_decoder(stream_->codecpar->codec_id);
    if (!codec) {
      throw std::runtime_error("No FFmpeg decoder for the video codec");
    }
    bool supports_cuda = false;
    for (int index = 0;; ++index) {
      const AVCodecHWConfig* config = avcodec_get_hw_config(codec, index);
      if (!config) {
        break;
      }
      if ((config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX) &&
          config->device_type == AV_HWDEVICE_TYPE_CUDA &&
          config->pix_fmt == AV_PIX_FMT_CUDA) {
        supports_cuda = true;
        break;
      }
    }
    if (!supports_cuda) {
      throw std::runtime_error("The selected FFmpeg decoder does not expose CUDA frames");
    }

    // CUDA runtime, TensorRT, and CUDA/GL use the device primary context.
    // A separate FFmpeg context would make its CUdeviceptrs invalid there.
    require(av_hwdevice_ctx_create(&hardware_device_, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr,
                                   AV_CUDA_USE_PRIMARY_CONTEXT),
            "create FFmpeg CUDA device");
    codec_ = avcodec_alloc_context3(codec);
    if (!codec_) {
      throw std::bad_alloc();
    }
    require(avcodec_parameters_to_context(codec_, stream_->codecpar), "copy codec parameters");
    codec_->opaque = this;
    codec_->get_format = select_cuda_format;
    codec_->hw_device_ctx = av_buffer_ref(hardware_device_);
    require(avcodec_open2(codec_, codec, nullptr), "open CUDA decoder");

    frame_ = av_frame_alloc();
    packet_ = av_packet_alloc();
    if (!frame_ || !packet_) {
      throw std::bad_alloc();
    }
    const AVRational rate = av_guess_frame_rate(format_, stream_, nullptr);
    nominal_fps_ = rate.den ? av_q2d(rate) : 0.0;
    codec_name_ = codec->name;
    frame_index_ = 0;
    draining_ = false;
  }

  bool next(GpuFrame& output) override {
    for (;;) {
      const int receive = avcodec_receive_frame(codec_, frame_);
      if (receive == 0) {
        if (frame_->format != AV_PIX_FMT_CUDA || !frame_->hw_frames_ctx) {
          throw std::runtime_error("Decoder returned a host frame; zero-copy CUDA path is unavailable");
        }
        const auto* frames = reinterpret_cast<const AVHWFramesContext*>(frame_->hw_frames_ctx->data);
        PixelFormat pixel_format;
        if (frames->sw_format == AV_PIX_FMT_NV12) {
          pixel_format = PixelFormat::nv12;
        } else if (frames->sw_format == AV_PIX_FMT_P010LE) {
          pixel_format = PixelFormat::p010;
        } else {
          const char* name = av_get_pix_fmt_name(frames->sw_format);
          throw std::runtime_error(std::string("Unsupported NVDEC surface format: ") +
                                   (name ? name : "unknown"));
        }
        const std::int64_t timestamp = frame_->best_effort_timestamp == AV_NOPTS_VALUE
                                           ? 0
                                           : frame_->best_effort_timestamp;
        output = {
            .frame_index = frame_index_++,
            .timestamp_ns = av_rescale_q(timestamp, stream_->time_base, AVRational{1, 1000000000}),
            .pts = frame_->pts == AV_NOPTS_VALUE ? timestamp : frame_->pts,
            .width = frame_->width,
            .height = frame_->height,
            .format = pixel_format,
            .luma = reinterpret_cast<std::uintptr_t>(frame_->data[0]),
            .chroma = reinterpret_cast<std::uintptr_t>(frame_->data[1]),
            .luma_pitch = static_cast<std::size_t>(frame_->linesize[0]),
            .chroma_pitch = static_cast<std::size_t>(frame_->linesize[1]),
        };
        return true;
      }
      if (receive == AVERROR_EOF) {
        return false;
      }
      if (receive != AVERROR(EAGAIN)) {
        require(receive, "decode frame");
      }
      if (draining_) throw std::runtime_error("Decoder requested input after the drain packet");

      for (;;) {
        if (!packet_->buf && packet_->size == 0) {
          const int read = av_read_frame(format_, packet_);
          if (read == AVERROR_EOF) {
            require(avcodec_send_packet(codec_, nullptr), "drain decoder");
            draining_ = true;
            break;
          }
          require(read, "read packet");
          if (packet_->stream_index != stream_index_) {
            av_packet_unref(packet_);
            continue;
          }
        }
        const int send = avcodec_send_packet(codec_, packet_);
        // EAGAIN means receive the pending decoded frame, then retry this
        // same packet. Dropping it would silently skip video frames.
        if (send == AVERROR(EAGAIN)) break;
        av_packet_unref(packet_);
        require(send, "submit packet");
        break;
      }
    }
  }

  [[nodiscard]] double nominal_fps() const override { return nominal_fps_; }
  [[nodiscard]] std::string codec_name() const override { return codec_name_; }

 private:
  static AVPixelFormat select_cuda_format(AVCodecContext*, const AVPixelFormat* formats) {
    for (const AVPixelFormat* current = formats; *current != AV_PIX_FMT_NONE; ++current) {
      if (*current == AV_PIX_FMT_CUDA) {
        return *current;
      }
    }
    return AV_PIX_FMT_NONE;
  }

  void close() {
    av_packet_free(&packet_);
    av_frame_free(&frame_);
    avcodec_free_context(&codec_);
    av_buffer_unref(&hardware_device_);
    if (format_) {
      avformat_close_input(&format_);
    }
    stream_ = nullptr;
    stream_index_ = -1;
  }

  AVFormatContext* format_{};
  AVCodecContext* codec_{};
  AVBufferRef* hardware_device_{};
  AVFrame* frame_{};
  AVPacket* packet_{};
  AVStream* stream_{};
  int stream_index_{-1};
  std::uint64_t frame_index_{};
  double nominal_fps_{};
  std::string codec_name_;
  bool draining_{};
};

}  // namespace

std::unique_ptr<VideoDecoder> make_ffmpeg_cuda_decoder() {
  return std::make_unique<FfmpegCudaDecoder>();
}

}  // namespace slam_native
