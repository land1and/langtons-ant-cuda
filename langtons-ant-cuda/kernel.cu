#include <windows.h>
#include <string>
#include <random>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define START_PATTERN 0 // pattern to start from
#define BATCH_SIZE 100 // patterns per batch
#define BATCH_COUNT 1 // number of batches
#define INVERT_PATTERN false // reverse and flip the pattern, for example turning "101011" into "001010"
#define MAX_ITERATIONS 100000000 // max iterations

#define GRID_SIZE (1024 & ~3) // size of the grid and resulting image rounded down to the neaest multiple of 4
#define GRID_SIZE_HALF (GRID_SIZE / 2)
#define GRID_SQUARED (GRID_SIZE * GRID_SIZE)
#define GRID_INDEX (GRID_SIZE * GRID_SIZE_HALF) + GRID_SIZE_HALF
#define FILE_SIZE (GRID_SQUARED + 310)

static const uint8_t bmp_header[54] = {
    0x42, 0x4D, // signature
    FILE_SIZE & 0xFF, (FILE_SIZE >> 8) & 0xFF, (FILE_SIZE >> 16) & 0xFF, (FILE_SIZE >> 24) & 0xFF, // file size
    0x00, 0x00, 0x00, 0x00, // reserved
    0x36, 0x01, 0x00, 0x00, // offset
    0x28, 0x00, 0x00, 0x00, // header size
    GRID_SIZE & 0xFF, (GRID_SIZE >> 8) & 0xFF, (GRID_SIZE >> 16) & 0xFF, (GRID_SIZE >> 24) & 0xFF, // width
    GRID_SIZE & 0xFF, (GRID_SIZE >> 8) & 0xFF, (GRID_SIZE >> 16) & 0xFF, (GRID_SIZE >> 24) & 0xFF, // height
    0x01, 0x00, // color planes
    0x08, 0x00, // bits per pixel
    0x00, 0x00, 0x00, 0x00, // compression
    GRID_SQUARED & 0xFF, (GRID_SQUARED >> 8) & 0xFF, (GRID_SQUARED >> 16) & 0xFF, (GRID_SQUARED >> 24) & 0xFF, // image size
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // ppm resolution
    0x40, 0x00, 0x00, 0x00, // number of colors
    0x00, 0x00, 0x00, 0x00 // important colors size
};

void save_bmp(const uint8_t* grid, const uint8_t* palette, const std::string& filename) {
    HANDLE handle = CreateFileA(filename.c_str(), GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (handle == INVALID_HANDLE_VALUE) return;
    DWORD written;
    WriteFile(handle, bmp_header, 54, &written, NULL);
    WriteFile(handle, palette, 256, &written, NULL);
    WriteFile(handle, grid, GRID_SQUARED, &written, NULL);
    CloseHandle(handle);
}

__launch_bounds__(1)
__global__ void ant_kernel(const uint8_t* __restrict__ sizes, const uint8_t* __restrict__ patterns, uint8_t* __restrict__ grids) {
    const uint32_t i = blockIdx.x;
    const uint8_t size = sizes[i];
    const uint8_t* pattern = &patterns[i * 64];
    uint8_t* grid = &grids[(uint64_t)i * GRID_SQUARED];
    uint32_t index = GRID_INDEX;
    uint32_t ant_position_x = GRID_SIZE_HALF;
    uint32_t ant_position_y = GRID_SIZE_HALF;
    int32_t ant_direction = 1;
    uint32_t state = 0;
    for (uint64_t j = 0; j < MAX_ITERATIONS; j += 2) {
        state = grid[index];
        grid[index] = state < size ? state + 1 : 0;
        if (pattern[state]) ant_direction = -ant_direction;
        ant_position_x += ant_direction;
        if (ant_position_x >= GRID_SIZE) break;
        index = ant_position_y * GRID_SIZE + ant_position_x;
        state = grid[index];
        grid[index] = state < size ? state + 1 : 0;
        if (!pattern[state]) ant_direction = -ant_direction;
        ant_position_y += ant_direction;
        if (ant_position_y >= GRID_SIZE) break;
        index = ant_position_y * GRID_SIZE + ant_position_x;
    }
}

int main() {
    std::mt19937 gen{ std::random_device{}() };
    std::uniform_int_distribution<> dist(0, 255);
    for (uint64_t batch = 0; batch < BATCH_COUNT; ++batch) {
        const uint64_t start_pattern = START_PATTERN + (batch * BATCH_SIZE);
        const uint64_t end_pattern = start_pattern + BATCH_SIZE;
        uint64_t* valid_patterns = new uint64_t[BATCH_SIZE];
        uint64_t num_valid_patterns = 0;
        for (uint64_t i = start_pattern; i < end_pattern; ++i) {
            if (!((i + 1) & i)) continue;
            valid_patterns[num_valid_patterns++] = i;
        }
        uint8_t* sizes = new uint8_t[num_valid_patterns];
        uint8_t* patterns = new uint8_t[num_valid_patterns * 64]();
        uint8_t* palettes = new uint8_t[num_valid_patterns * 256]();
        uint8_t* grids = new uint8_t[num_valid_patterns * GRID_SQUARED]();
        for (uint64_t i = 0; i < num_valid_patterns; ++i) {
            const uint64_t pattern = valid_patterns[i];
            uint8_t size_minus_one = 0;
            for (int8_t j = 63; j >= 0; --j) {
                if (pattern & (1ULL << j)) {
                    size_minus_one = j;
                    break;
                }
            }
            const uint8_t size = size_minus_one + 1;
            sizes[i] = size - 1;
            const uint64_t i_64 = i * 64;
            for (uint8_t j = 0; j < size; ++j) {
                patterns[i_64 + (INVERT_PATTERN ? j : size_minus_one - j)] = ((pattern >> j) & 1) ^ INVERT_PATTERN;
            }
            const uint64_t i_256 = i * 256;
            const uint8_t size_4_1 = (size * 4) - 1;
            for (uint8_t j = 0; j < size_4_1; ++j) {
                palettes[i_256 + j] = dist(gen);
            }
        }
        uint8_t* d_sizes;
        uint8_t* d_patterns;
        uint8_t* d_grids;
        cudaMalloc(&d_sizes, num_valid_patterns);
        cudaMalloc(&d_patterns, num_valid_patterns * 64);
        cudaMalloc(&d_grids, num_valid_patterns * GRID_SQUARED);
        cudaMemcpy(d_sizes, sizes, num_valid_patterns, cudaMemcpyHostToDevice);
        cudaMemcpy(d_patterns, patterns, num_valid_patterns * 64, cudaMemcpyHostToDevice);
        cudaMemcpy(d_grids, grids, num_valid_patterns * GRID_SQUARED, cudaMemcpyHostToDevice);
        ant_kernel << <num_valid_patterns, 1 >> > (d_sizes, d_patterns, d_grids);
        cudaDeviceSynchronize();
        cudaMemcpy(grids, d_grids, num_valid_patterns * GRID_SQUARED, cudaMemcpyDeviceToHost);
        cudaFree(d_sizes);
        cudaFree(d_patterns);
        cudaFree(d_grids);
        for (uint64_t i = 0; i < num_valid_patterns; ++i) {
            save_bmp(&grids[i * GRID_SQUARED], &palettes[i * 256], std::to_string(valid_patterns[i]) + ".bmp");
        }
        delete[] valid_patterns;
        delete[] sizes;
        delete[] patterns;
        delete[] palettes;
        delete[] grids;
    }
}