cmake_minimum_required(VERSION 3.13)

# Pull in Pico SDK (must be set before project())

set(PICO_SDK_PATH ~/ResNET-PICO/pico-sdk)
include(~/ResNET-PICO/pico-sdk/external/pico_sdk_import.cmake)

project(resnet_pico C CXX ASM)
set(CMAKE_C_STANDARD 11)

pico_sdk_init()

add_executable(resnet_pico
    resnet_pico.c
)

target_compile_definitions(resnet_pico PRIVATE PICO_BUILD=1)

# Maximise optimisation for speed on Cortex-M0+
target_compile_options(resnet_pico PRIVATE -O3 -ffast-math)

target_link_libraries(resnet_pico
    pico_stdlib
    hardware_watchdog
)

# Enable USB serial output (printf)
pico_enable_stdio_usb(resnet_pico 1)
pico_enable_stdio_uart(resnet_pico 0)

# Create .uf2 and other artifacts
pico_add_extra_outputs(resnet_pico)
