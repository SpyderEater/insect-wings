#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>

namespace py = pybind11;

void process_image_cpp(py::array_t<uint8_t> input, py::array_t<uint8_t> output, int radius, int threshold_val) {
    auto buf_in = input.request();
    auto buf_out = output.request();

    const uint8_t* __restrict__ ptr_in = static_cast<const uint8_t*>(buf_in.ptr);
    uint8_t* __restrict__ ptr_out = static_cast<uint8_t*>(buf_out.ptr);

    const int height = (int)buf_in.shape[0];
    const int width = (int)buf_in.shape[1];
    const float window_size = (float)((2 * radius + 1) * (2 * radius + 1));

    tbb::parallel_for(tbb::blocked_range2d<int>(radius, height - radius, radius, width - radius),
        [&](const tbb::blocked_range2d<int>& r) {
            
            for (int y = r.rows().begin(); y < r.rows().end(); ++y) {
                const int row_offset = y * width;
                
                for (int x = r.cols().begin(); x < r.cols().end(); ++x) {
                    int sum = 0;

                    for (int ky = -radius; ky <= radius; ++ky) {
                        const int neighbor_row = (y + ky) * width;
                        for (int kx = -radius; kx <= radius; ++kx) {
                            sum += ptr_in[neighbor_row + (x + kx)];
                        }
                    }

                    ptr_out[row_offset + x] = ((float)sum / window_size > threshold_val) ? 255 : 0;
                }
            }
        });
}

PYBIND11_MODULE(image_processor, m) {
    m.def("process_image_cpp", &process_image_cpp, "High-speed TBB image processor",
          py::arg("input"), py::arg("output"), py::arg("radius"), py::arg("threshold_val"));
}