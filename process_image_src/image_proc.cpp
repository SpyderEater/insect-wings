#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>
#include <vector>
#include <algorithm>

namespace py = pybind11;

void process_image_cpp(py::array_t<uint8_t> input, py::array_t<uint8_t> output, int radius, int threshold_val) {
    auto buf_in = input.request();
    auto buf_out = output.request();

    uint8_t* ptr_in = static_cast<uint8_t*>(buf_in.ptr);
    uint8_t* ptr_out = static_cast<uint8_t*>(buf_out.ptr);

    int height = buf_in.shape[0];
    int width = buf_in.shape[1];

    tbb::parallel_for(tbb::blocked_range2d<int>(radius, height - radius, radius, width - radius),
        [&](const tbb::blocked_range2d<int>& r) {
            
            std::vector<uint8_t> window;
            window.reserve((2 * radius + 1) * (2 * radius + 1));

            for (int y = r.rows().begin(); y < r.rows().end(); ++y) {
                for (int x = r.cols().begin(); x < r.cols().end(); ++x) {
                    
                    window.clear();
                    
                    for (int ky = -radius; ky <= radius; ++ky) {
                        for (int kx = -radius; kx <= radius; ++kx) {
                            window.push_back(ptr_in[(y + ky) * width + (x + kx)]);
                        }
                    }

                    int sum = 0;
                    for (uint8_t val : window) sum += val;
                    int avg = sum / window.size();

                    ptr_out[y * width + x] = (avg > threshold_val) ? 255 : 0;
                }
            }
        });
}

PYBIND11_MODULE(image_processor, m) {
    m.def("process_image_cpp", &process_image_cpp, "A function to process image pixels using TBB",
          py::arg("input"), py::arg("output"), py::arg("radius"), py::arg("threshold_val"));
}