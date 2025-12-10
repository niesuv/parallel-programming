add_executable(test_data_loader test_data_loader.cpp ../src/data_loader.cpp)

add_executable(train_cpu train_cpu.cpp ../src/data_loader.cpp ../src/cpu_layers.cpp ../src/autoencoder.cpp)

target_include_directories(test_data_loader PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(train_cpu PRIVATE ${PROJECT_SOURCE_DIR}/include)
