
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <algorithm>
#include <chrono>
#include <iostream>

//#include <json.hpp>
#include <cxxopts.hpp>
#include <rapidjson/rapidjson.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/ostreamwrapper.h>
#include <algorithm> 

constexpr char KEY_CONFIG[] = "config";
constexpr char KEY_DURATION[] = "duration";
constexpr char KEY_DURATION_TYPE[] = "duration-type";
constexpr char KEY_HEIGHT[] = "height";
constexpr char KEY_HELP[] = "help";
constexpr char KEY_OUTPUT[] = "output";
constexpr char KEY_TYPE[] = "type";
constexpr char KEY_WIDTH[] = "width";

class cLimitTimer
{
    std::chrono::time_point<std::chrono::steady_clock> Start;
    double Limit;
public:
    cLimitTimer(double limit) : Limit(limit)
    {
        Start = std::chrono::steady_clock::now();
    }
    ~cLimitTimer()
    {
        auto end = std::chrono::steady_clock::now();

        std::chrono::duration<double> diff = end - Start;
        if (diff.count() > Limit)
            std::cout << diff.count() << "s greater than limit " << Limit << "s\n";
        else if(diff.count() > 1.0)
            std::cout << diff.count() << "s\n";
        else
            std::cout << diff.count() * 1000 << "ms\n";

    }
};




__global__ void addKernel(int *B, int *A, unsigned int size)
{
    //1;
    for (int i = 0; i < size; i++)
         B[i] = A[i];
}


namespace application {
enum class Type {
    Invalid,
    Floating,
    Integer
};

Type stringToType(const std::string& key)
{
    Type result = Type::Invalid;

    std::map<std::string, Type> mapOfType = {
        {"floating", Type::Floating},
        {"integer", Type::Integer}
    };

    std::map<std::string, Type>::iterator it = mapOfType.find(key);

    if (it != mapOfType.end())
        result = it->second;
    
    return result;
}

enum class DurationType {
    Invalid,
    Counter,
    Timer
};


DurationType stringToDurationType(const std::string& key)
{
    DurationType result = DurationType::Invalid;

    std::map<std::string, DurationType> mapOfDurationType = {
        {"counter", DurationType::Counter},
        {"timer", DurationType::Timer}
    };

    std::map<std::string, DurationType>::iterator it = mapOfDurationType.find(key);

    if (it != mapOfDurationType.end())
        result = it->second;

    return result;
}

enum class Output {
    Invalid,
    File,
    Screen
};

Output stringToOutput(const std::string& key)
{
    Output result = Output::Invalid;

    std::map<std::string, Output> mapOfOutput = {
        {"file", Output::File},
        {"screen", Output::Screen}
    };

    std::map<std::string, Output>::iterator it = mapOfOutput.find(key);

    if (it != mapOfOutput.end())
        result = it->second;

    return result;
}

struct Parameters {
    int duration = -1;
    DurationType durationType = DurationType::Invalid;
    int height = -1;
    Output output = Output::Invalid;
    Type type = Type::Invalid;
    int width = -1;
};
}

int readJsonFile(const std::string configFile, application::Parameters &parameters)
{
    using namespace rapidjson;

    std::ifstream ifs{ configFile };
    if (!ifs.is_open())
    {
        std::cerr << "Could not open " << configFile << " file!" << std::endl;
        return 1;
    }

    IStreamWrapper isw{ ifs };

    Document doc{};
    doc.ParseStream(isw);

    if (doc.HasParseError())
    {
        std::cerr << "Error  : " << doc.GetParseError() << '\n'
            << "Offset : " << doc.GetErrorOffset() << std::endl;
        return 1;
    }

    if (doc.HasMember(KEY_DURATION) && doc[KEY_DURATION].IsInt())
        parameters.duration = doc[KEY_DURATION].GetInt();

    if (doc.HasMember(KEY_DURATION_TYPE) && doc[KEY_DURATION_TYPE].IsString())
    {
        auto durationType = doc[KEY_DURATION_TYPE].GetString();
        parameters.durationType = application::stringToDurationType(durationType);
    }

    if (doc.HasMember(KEY_HEIGHT) && doc[KEY_HEIGHT].IsInt())
        parameters.height = doc[KEY_HEIGHT].GetInt();

    if (doc.HasMember(KEY_TYPE) && doc[KEY_TYPE].IsString())
    {
        auto type = doc[KEY_TYPE].GetString();
        parameters.type = application::stringToType(type);
    }

    if (doc.HasMember(KEY_OUTPUT) && doc[KEY_OUTPUT].IsString())
    {
        auto output = doc[KEY_OUTPUT].GetString();
        parameters.output = application::stringToOutput(output);
    }

    if (doc.HasMember(KEY_WIDTH) && doc[KEY_WIDTH].IsInt())
        parameters.width = doc[KEY_WIDTH].GetInt();

    return 0;
}

int validate(application::Parameters& parameters)
{
    int err = 0;

    if (parameters.duration < 1)
    {
        std::cerr << "Invalid duration!" << std::endl;
        err = 1;
    } 
    else if (parameters.durationType == application::DurationType::Invalid)
    {
        std::cerr << "Invalid duration type" << std::endl;
        err = 1;
    }
    else if (parameters.height < 1)
    {
        std::cerr << "Invalid height!" << std::endl;
        err = 1;
    }
    else if (parameters.output == application::Output::Invalid)
    {
        std::cerr << "Invalid output type" << std::endl;
        err = 1;
    }
    else if (parameters.type == application::Type::Invalid)
    {
        std::cerr << "Invalid type" << std::endl;
        err = 1;
    }
    else if (parameters.width< 1)
    {
        std::cerr << "Invalid width!" << std::endl;
        err = 1;
    }

    return err;
}

int applyOptions(cxxopts::ParseResult &result, application::Parameters& parameters)
{
    int err = 0;
    if (result.count(KEY_CONFIG))
    {
        std::string configFile = { result[KEY_CONFIG].as<std::string>() };
        err = readJsonFile(configFile, parameters);
    }

    if (err == 0)
    {
        if (result.count(KEY_DURATION))
            parameters.duration = result[KEY_DURATION].as<int>();

        if (result.count(KEY_DURATION_TYPE))
        {
            auto durationType = result[KEY_DURATION_TYPE].as<std::string>();
            parameters.durationType = application::stringToDurationType(durationType);
        }

        if (result.count(KEY_HEIGHT))
            parameters.height = result[KEY_HEIGHT].as<int>();

        if (result.count(KEY_OUTPUT))
        {
            auto output = result[KEY_OUTPUT].as<std::string>();
            parameters.output = application::stringToOutput(output);
        }

        if (result.count(KEY_TYPE))
        {
            auto type = result[KEY_TYPE].as<std::string>();
            parameters.type = application::stringToType(type);
        }

        if (result.count(KEY_WIDTH))
            parameters.width = result[KEY_WIDTH].as<int>();
    }

    return err;
}

cudaError_t copyCuda(const application::Parameters& parameters, int*** arrayRGB);

void startCounterLoop(const application::Parameters& parameters, int*** arrayRGB)
{
    const int height = parameters.height;
    const int width = parameters.width;
    const int duration = parameters.duration;

    int h = 0;
    int w = 0;
    int c = 0;

    for (h = 0; h < height; h++) {
        printf("Height %d\n", h);
        for (w = 0; w < width; w++) {
            for (c = 0; c < 3; c++) {
                printf("%.2d ", arrayRGB[h][w][c]);
            }
            printf("\n");
        }
        printf("\n");
    }

    for (int i = 0; i < duration; i++)
    {
        //std::cout << "it: " << i << std::endl;
        copyCuda(parameters, arrayRGB);
    }

}

void startTimerLoop(application::Parameters& parameters, int*** arrayRGB)
{
    const int duration = parameters.duration;
    auto start = std::chrono::system_clock::now();
    auto end = std::chrono::system_clock::now();
    while ((std::chrono::duration_cast<std::chrono::seconds>(end - start).count() != duration))
    {
        copyCuda(parameters, arrayRGB);
        end = std::chrono::system_clock::now();
    }
}



void startCopyTest(application::Parameters &parameters)
{
    const int height = parameters.height;
    const int width = parameters.width;
    const int channels = 3; // add to options list
   
    int*** arrayRGB;


    int h = 0;
    int w = 0;
    int c = 0;

    arrayRGB = (int***)malloc(sizeof(int***) * height);
    for (h = 0; h < height; h++) {
        arrayRGB[h] = (int**)malloc(sizeof(int*) * width);
        for (w = 0; w < width; w++) {
            arrayRGB[h][w] = (int*)malloc(sizeof(int) * channels);
            for (c = 0; c < 3; c++)
                arrayRGB[h][w][c] = 6;
        }
    }

    if (parameters.durationType == application::DurationType::Counter)
        startCounterLoop(parameters, arrayRGB);
    else if (parameters.durationType == application::DurationType::Timer)
        startTimerLoop(parameters, arrayRGB);


    for (h = 0; h < height; h++)  {
        for (w = 0; w < width; w++) {
            free(arrayRGB[h][w]);
        }
        free(arrayRGB[h]);
    }
    free(arrayRGB);

}

int main(int argc, char** argv)
{
    application::Parameters parameters;

    cxxopts::ParseResult result;
    cxxopts::Options options("first-CUDA-Project", "copy back and forth from host to GPU an array and ");

    options.add_options()
        (KEY_CONFIG, "Config file to load.", cxxopts::value<std::string>()->default_value(""))
        (KEY_DURATION, "Number of iterations or time in seconds.", cxxopts::value<int>())
        (KEY_DURATION_TYPE, "Possible value is counter or timer.", cxxopts::value<std::string>())
        (KEY_HEIGHT, "Height of the array.", cxxopts::value<int>())
        (KEY_OUTPUT, "Possible value is screen or file.", cxxopts::value<std::string>())
        (KEY_TYPE, "Possible value is floating or integer.", cxxopts::value<std::string>())
        (KEY_WIDTH, "Width of the array", cxxopts::value<int>())
        (KEY_HELP, "Print usage.");

    try
    {
        result = options.parse(argc, argv);
    }
    catch (...)
    {
        std::cerr << "Exception thrown from parse!" << std::endl;
    }

    if (result.count("help"))
        std::cout << options.help() << std::endl;
    else if (applyOptions(result, parameters))
        std::cerr << "(EE) applyOptions -> error" << std::endl;
    else if (validate(parameters))
        std::cerr << "(EE) validate -> error" << std::endl;
    else
        startCopyTest(parameters);

    std::cout << "duration: " << parameters.duration << std::endl;
    std::cout << "duration-type: " << (parameters.durationType == application::DurationType::Timer ? "Timer" : "Counter") << std::endl;
    std::cout << "height: " << parameters.height << std::endl;
    std::cout << "output: " << (parameters.output == application::Output::File ? "File" : "Screen") << std::endl;
    std::cout << "type: " << (parameters.type == application::Type::Floating ? "Floating" : "Integer") << std::endl;
    std::cout << "width: " << parameters.width << std::endl;
    
   return 0;
}

int to1D(const int h, const int w, const int c, const application::Parameters& parameters) {
    const int height = parameters.height;
    const int width = parameters.width;

    return (c * height * width) + (w * height) + h;
}

cudaError_t copyCuda(const application::Parameters& parameters, int*** arrayRGB)
{
    const int height = parameters.height;
    const int width = parameters.width;
    const int channels = 3; // add to options list


    auto* timer1 = new cLimitTimer(1000);
    cudaError_t cudaStatus;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    float milliseconds = 0;
    float milliseconds2 = 0;
    int* hostToGpu1D = (int*)malloc(sizeof(int) * height * height * channels);

    int* GpuToHost1D = (int*)malloc(sizeof(int) * height * height * channels);

    int* onGpu = 0;

    printf("initialization time: ");
    delete timer1;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0); // add this to options
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;

    }

    // Allocate GPU buffers for three vectors (two input, one output).
    cudaStatus = cudaMalloc((void**)&onGpu, height * width * channels * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    timer1 = new cLimitTimer(1000);
    cudaEventRecord(start);
    cudaStatus = cudaMemcpy(onGpu, hostToGpu1D, height * width * channels * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy onGpu, hostToGpu1D!");
        goto Error;
    }

    cudaEventRecord(stop);
    printf("Host->GPU... : ");
    delete timer1;

    // <Launch a kernel on the GPU here>

    // Copy output vector from GPU buffer to host memory.
    timer1 = new cLimitTimer(1000);
    cudaEventRecord(start2);
    cudaStatus = cudaMemcpy(GpuToHost1D, onGpu, height * width * channels * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy GpuToHost1D, onGpu failed!");
        goto Error;
    }

    cudaEventRecord(stop2);
    printf("GPU->Host... : ");
    delete timer1;

    cudaEventSynchronize(stop);
    cudaEventSynchronize(stop2);

    printf("\n\nCuda event timers:\n");
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Host->GPU, Ellapsed time: %fms\n", milliseconds);
    cudaEventElapsedTime(&milliseconds2, start2, stop2);
    printf("GPU->Host, Ellapsed time: %fms\n", milliseconds2);

Error:
    cudaFree(onGpu);

    return cudaStatus;
}
