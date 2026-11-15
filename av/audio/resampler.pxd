from libc.stdint cimport uint64_t

from av.audio.format cimport AudioFormat
from av.audio.frame cimport AudioFrame
from av.audio.layout cimport AudioLayout
from av.filter.graph cimport Graph


cdef class AudioResampler:
    cdef AudioFrame template
    cdef Graph graph
    cdef readonly AudioFormat format
    cdef readonly AudioLayout layout
    cdef readonly dict options
    cdef readonly int rate
    cdef readonly unsigned int frame_size
    cdef readonly int threads
    cdef readonly bint is_passthrough

    cdef object input_time_base
    cdef object input_pts_anchor
    cdef object output_pts_anchor
    cdef uint64_t input_samples
    cdef uint64_t output_samples

    cpdef list[AudioFrame] resample(self, AudioFrame)
