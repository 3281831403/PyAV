cimport libav as lib

from av.filter.filter cimport Filter
from av.filter.graph cimport Graph
from av.frame cimport Frame


cdef class FilterContext:
    cdef lib.AVFilterContext *ptr
    cdef readonly Graph graph
    cdef readonly Filter filter
    cdef bint inited
    cdef unsigned char _kind

    cdef void _push_source_frame(self, Frame frame)


cdef FilterContext wrap_filter_context(Graph graph, Filter filter, lib.AVFilterContext *ptr)
