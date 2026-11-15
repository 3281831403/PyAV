from errno import EAGAIN
from fractions import Fraction

import cython
from cython.cimports.av.filter.graph import Graph

from av.error import FFmpegError


@cython.final
@cython.cclass
class AudioResampler:
    """AudioResampler(format=None, layout=None, rate=None, frame_size=None, options=None, threads=None)

    :param AudioFormat format: The target format, or string that parses to one
        (e.g. ``"s16"``).
    :param AudioLayout layout: The target layout, or an int/string that parses
        to one (e.g. ``"stereo"``).
    :param int rate: The target sample rate.
    :param int frame_size: The number of samples per output frame.
    :param dict options: ``libswresample`` options passed to the underlying
        ``aresample`` filter (e.g. ``{"resampler": "soxr", "precision": "28"}``).
        See the `FFmpeg resampler documentation
        <https://ffmpeg.org/ffmpeg-resampler.html>`_ for the full list.
    :param int threads: How many threads the filter graph may use, or ``0`` for
        automatic selection based on the number of available CPUs. Defaults to
        ``0`` (auto).
    """

    def __cinit__(
        self,
        format=None,
        layout=None,
        rate=None,
        frame_size=None,
        options=None,
        threads=None,
    ):
        if format is not None:
            self.format = (
                format if isinstance(format, AudioFormat) else AudioFormat(format)
            )

        if layout is not None:
            self.layout = AudioLayout(layout)

        self.rate = int(rate) if rate else 0
        self.frame_size = int(frame_size) if frame_size else 0
        self.options = {str(k): str(v) for k, v in options.items()} if options else {}
        self.threads = int(threads) if threads else 0
        self.graph = None

    @cython.ccall
    def resample(self, frame: AudioFrame | None) -> list[AudioFrame]:
        """resample(frame)

        Convert the ``sample_rate``, ``channel_layout`` and/or ``format`` of
        a :class:`~.AudioFrame`.

        :param AudioFrame frame: The frame to convert or `None` to flush.
        :returns: A list of :class:`AudioFrame` in new parameters. If the nothing is to be done return the same frame
            as a single element list.

        """
        # We don't have any input, so don't bother even setting up.
        if not self.graph and frame is None:
            return []

        # Shortcut for passthrough.
        if self.is_passthrough:
            return [frame]

        # Take source settings from the first frame.
        if not self.graph:
            self.template = frame
            self.input_time_base = frame.time_base

            # Set some default descriptors.
            self.format = self.format or frame.format
            self.layout = self.layout or frame.layout
            self.rate = self.rate or frame.sample_rate

            # Check if we can passthrough or if there is actually work to do.
            if (
                frame.format.sample_fmt == self.format.sample_fmt
                and frame.layout == self.layout
                and frame.sample_rate == self.rate
                and self.frame_size == 0
            ):
                self.is_passthrough = True
                return [frame]

            # handle resampling with aformat filter
            # (similar to configure_output_audio_filter from ffmpeg)
            self.graph = Graph()
            self.graph.threads = self.threads
            extra_args = {}
            if frame.time_base:
                extra_args["time_base"] = f"{frame.time_base}"

            abuffer = self.graph.add(
                "abuffer",
                sample_rate=f"{frame.sample_rate}",
                sample_fmt=AudioFormat(frame.format).name,
                channel_layout=frame.layout.name,
                **extra_args,
            )
            aformat = self.graph.add(
                "aformat",
                sample_rates=f"{self.rate}",
                sample_fmts=self.format.name,
                channel_layouts=self.layout.name,
            )
            abuffersink = self.graph.add("abuffersink")

            # When libswresample options are given, do the conversion with an
            # explicit aresample filter (which owns the SwrContext) instead of
            # relying on the one FFmpeg auto-inserts before aformat.
            if self.options:
                aresample = self.graph.add("aresample", **self.options)
                abuffer.link_to(aresample)
                aresample.link_to(aformat)
            else:
                abuffer.link_to(aformat)

            aformat.link_to(abuffersink)
            self.graph.configure()

            if self.frame_size > 0:
                self.graph.set_audio_frame_size(self.frame_size)

        if frame is not None:
            if (
                frame.format.sample_fmt != self.template.format.sample_fmt
                or frame.layout != self.template.layout
                or frame.sample_rate != self.template.rate
                or frame.time_base != self.input_time_base
            ):
                raise ValueError("Frame does not match AudioResampler setup.")

            # Track the input timeline so that output timestamps can advance
            # strictly by sample count. Frames from a source with a time_base
            # coarser than the sample rate (e.g. 1/1000 for 44.1khz audio)
            # carry individually rounded timestamps, which must not be allowed
            # to make the output timeline drift.
            if frame.pts is not None:
                time_base = frame.time_base or Fraction(1, frame.sample_rate)
                expected_pts = self.input_pts_anchor
                if expected_pts is not None:
                    expected_pts += round(
                        Fraction(self.input_samples, frame.sample_rate) / time_base
                    )
                if expected_pts is None or abs(frame.pts - expected_pts) > 1:
                    # The first frame with a pts (which may be negative, e.g.
                    # AAC priming samples), or a real gap in the input: anchor
                    # both timelines on it.
                    self.input_pts_anchor = frame.pts - round(
                        Fraction(self.input_samples, frame.sample_rate) / time_base
                    )
                    self.output_pts_anchor = round(
                        self.input_pts_anchor * time_base * self.rate
                    )

            self.input_samples += frame.samples

        self.graph.push(frame)

        output: list[AudioFrame] = []
        while True:
            try:
                oframe = self.graph.pull()
            except EOFError:
                break
            except FFmpegError as e:
                if e.errno != EAGAIN:
                    raise
                break

            if self.output_pts_anchor is not None:
                oframe.pts = self.output_pts_anchor + self.output_samples
            self.output_samples += oframe.samples
            output.append(oframe)

        return output
