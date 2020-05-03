# set source files
file(GLOB TARGET_SOURCES "src-seq/" *.cpp)

# add the executable
add_executable(rnn-seq ${TARGET_SOURCES})