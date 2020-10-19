
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
#include <list>
#include "nvToolsExt.h"
#include <limits> // press enter...

constexpr char KEY_CHANNEL[] = "channel";
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

__global__ void addKernel()
{
    // <kernel code here>
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

struct Parameters {
    int channel = -1;
    int duration = -1;
    DurationType durationType = DurationType::Invalid;
    int height = -1;
    std::string output {};
    Type type = Type::Invalid;
    int width = -1;
};

struct Timers {
    float timer1 = -1.0f;
    float timer2 = -1.0f;
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

    if (doc.HasMember(KEY_CHANNEL) && doc[KEY_CHANNEL].IsInt())
        parameters.channel = doc[KEY_CHANNEL].GetInt();

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
        parameters.output = doc[KEY_OUTPUT].GetString();
    }

    if (doc.HasMember(KEY_WIDTH) && doc[KEY_WIDTH].IsInt())
        parameters.width = doc[KEY_WIDTH].GetInt();

    return 0;
}

int validate(application::Parameters& parameters)
{
    int err = 0;


    if (parameters.channel < 1)
    {
        std::cerr << "Invalid channel!" << std::endl;
        err = 1;
    }
    else if (parameters.duration < 1)
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
        if (result.count(KEY_CHANNEL))
            parameters.channel = result[KEY_CHANNEL].as<int>();

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
            parameters.output = output;
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

template <class T>
int copyCuda(const application::Parameters& parameters, T* arrayOnCpu, application::Timers &timers);

template <class T>
void startCounterLoop(const application::Parameters& parameters, T* arrayOnCpu, std::list<application::Timers> &listOfTimers)
{
    const int duration = parameters.duration;

    for (int i = 0; i < duration; i++)
    {
        application::Timers timers;
        if (copyCuda(parameters, arrayOnCpu, timers))
            break;
        listOfTimers.push_back(timers);
    }
}

template <class T>
void startTimerLoop(application::Parameters& parameters, T* arrayOnCpu, std::list<application::Timers> & listOfTimers)
{
    const int duration = parameters.duration;

    auto start = std::chrono::system_clock::now();
    auto end = std::chrono::system_clock::now();
    while ((std::chrono::duration_cast<std::chrono::seconds>(end - start).count() != duration))
    {
        application::Timers timers;
        if (copyCuda(parameters, arrayOnCpu, timers))
            break;
        listOfTimers.push_back(timers);
        end = std::chrono::system_clock::now();
    }
}

void printToScreen(std::list<application::Timers> &listOfTimers)
{
    for (auto const& i : listOfTimers)
        std::cout << i.timer1 << ", " << i.timer2 << std::endl;
}

void printToFile(const std::string &output, std::list<application::Timers>& listOfTimers)
{
    std::ofstream outputFile(output);

    if (outputFile.is_open())
        for (auto const& i : listOfTimers)
            outputFile << i.timer1 << ", " << i.timer2 << std::endl;
    else
        std::cerr << "cannot open file!" << std::endl;
}

void print(application::Parameters& parameters, std::list<application::Timers>& listOfTimers)
{
    const auto output = parameters.output;

    if (output.size())
        printToFile(output, listOfTimers);
    else
        printToScreen(listOfTimers);
}

template <class T>
int startCopyTest(application::Parameters &parameters, std::list<application::Timers> &listOfTimers)
{
    const int height = parameters.height;
    const int width = parameters.width;
    const int channels = parameters.channel; 
    const int size = height * width * channels;

    T *arrayOnCpu = (T*)malloc(sizeof(T) * size);

    if (arrayOnCpu == NULL)
    {
        std::cerr << "arrayOnCpu allocation failed!" << std::endl;
        return 1;
    }

    if (parameters.durationType == application::DurationType::Counter)
        startCounterLoop<T>(parameters, arrayOnCpu, listOfTimers);
    else if (parameters.durationType == application::DurationType::Timer)
        startTimerLoop<T>(parameters, arrayOnCpu, listOfTimers);

    free(arrayOnCpu);
    return 0;
}

int startCopyTest2(application::Parameters& parameters, std::list<application::Timers>& listOfTimers)
{
    int err = 0;
    auto type = parameters.type;

    switch (type)
    {
        case application::Type::Floating:
            err = startCopyTest<float>(parameters, listOfTimers);
            break;
        case application::Type::Integer:
            err = startCopyTest<int>(parameters, listOfTimers);
            break;
        case application::Type::Invalid:
            std::cerr << "Invalid type!" << std::endl;
            err = 1;
            break;
    }

    return err;
}


void print(application::Parameters &parameters)
{
    std::cout << "channel: " << parameters.channel << std::endl;
    std::cout << "duration: " << parameters.duration << std::endl;
    std::cout << "duration-type: " << (parameters.durationType == application::DurationType::Timer ? "Timer" : "Counter") << std::endl;
    std::cout << "height: " << parameters.height << std::endl;
    std::cout << "output: " << parameters.output << std::endl;
    std::cout << "type: " << (parameters.type == application::Type::Floating ? "Floating" : "Integer") << std::endl;
    std::cout << "width: " << parameters.width << std::endl;
}
#include <Windows.h>

int main(int argc, char** argv)
{
    /*
    nvtxNameOsThread(1, "MAIN");
    nvtxRangePush(__FUNCTION__);
    std::cout << "1F" << std::endl;
    Sleep(100);
    nvtxRangePop();

    // zero the structure
    nvtxEventAttributes_t eventAttrib = { 0 };

    // set the version and the size information
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;

    static const uint32_t COLOR_GREEN = 0xFF00FF00;

    // configure the attributes.  0 is the default for all attributes.
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = COLOR_GREEN;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = __FUNCTION__ ":nvtxRangePushEx TOTO3";

    nvtxRangePushEx(&eventAttrib);
    std::cout << "3F" << std::endl;
    Sleep(100);

        nvtxRangePop();

    return 0;
    */

    application::Parameters parameters;
    std::list<application::Timers> listOfTimers;
    int err = 0;

    cxxopts::ParseResult result;
    cxxopts::Options options("first-CUDA-Project", "copy back and forth from host to GPU an array and measure times.");

    options.add_options()
        (KEY_CHANNEL, "Number of channels for the array.", cxxopts::value<int>())
        (KEY_CONFIG, "Config file to load.", cxxopts::value<std::string>())
        (KEY_DURATION, "Number of iterations or time in seconds.", cxxopts::value<int>())
        (KEY_DURATION_TYPE, "Possible value is counter or timer.", cxxopts::value<std::string>())
        (KEY_HEIGHT, "Height of the array.", cxxopts::value<int>())
        (KEY_OUTPUT, "Store to file instead of print to screen.", cxxopts::value<std::string>())
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
    else if (err = applyOptions(result, parameters), err)
        std::cerr << "(EE) applyOptions -> error" << std::endl;
    else if (err = validate(parameters), err)
        std::cerr << "(EE) validate -> error" << std::endl;
    else if (err = startCopyTest2(parameters, listOfTimers), err)
        std::cerr << "(EE) startCopyTest -> error" << std::endl;
    else
        print(parameters, listOfTimers);
        //print(parameters);
    
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    auto cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        std::cerr <<  "cudaDeviceReset failed!" << std::endl;
        err = 1;
    }

   // std::cout << "Press Enter to ContinueX";
//#undef max // for visual studio intellisense...
  //  std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

   return err;
}

template <class T>
int copyCuda(const application::Parameters& parameters, T* arrayOnCpu, application::Timers& timers)
{
    const int height = parameters.height;
    const int width = parameters.width;
    const int channels = parameters.channel;
    const int size = height * width * channels;

    cudaError_t cudaStatus;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    T* resultArrayFromGPU = (T*)malloc(sizeof(T) * size);
    T* arrayOnGpu = 0;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0); // add this to options
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output).
    cudaStatus = cudaMalloc((void**)&arrayOnGpu, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    //nvtxNameOsThread(0, "Main Thread");
    //nvtxRangePush(__FUNCTION__);
    //nvtxRangePush("cudaEventRecord start");
    // zero the structure
    nvtxEventAttributes_t eventAttrib = { 0 };

    // set the version and the size information
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;

    static const uint32_t COLOR_GREEN = 0xFF00FF00;

    // configure the attributes.  0 is the default for all attributes.
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = COLOR_GREEN;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = __FUNCTION__ ":timer1 ";

    nvtxRangePushEx(&eventAttrib);
    
    //Sleep(100);

    //nvtxRangePop();
    cudaEventRecord(start);
    cudaStatus = cudaMemcpy(arrayOnGpu, arrayOnCpu, size * sizeof(T), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy arrayOnGpu, hostToGpu1D!");
        goto Error;
    }
    cudaEventRecord(stop);
    nvtxRangePop();

    // <Launch a kernel on the GPU here>

    // Copy output vector from GPU buffer to host memory.
    //nvtxRangePushA("cudaEventRecord start2");
    // zero the structure
    nvtxEventAttributes_t eventAttrib2 = { 0 };

    // set the version and the size information
    eventAttrib2.version = NVTX_VERSION;
    eventAttrib2.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;

    // configure the attributes.  0 is the default for all attributes.
    eventAttrib2.colorType = NVTX_COLOR_ARGB;
    eventAttrib2.color = COLOR_GREEN;
    eventAttrib2.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib2.message.ascii = __FUNCTION__ ":timer2 ";

    nvtxRangePushEx(&eventAttrib2);
    cudaEventRecord(start2);
    cudaStatus = cudaMemcpy(resultArrayFromGPU, arrayOnGpu, size * sizeof(T), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy resultArrayFromGPU, arrayOnGpu failed!");
        goto Error;
    }
    cudaEventRecord(stop2);
    nvtxRangePop();

    cudaEventSynchronize(stop);
    cudaEventSynchronize(stop2);

    // Get elapsed times.
    cudaEventElapsedTime(&timers.timer1, start, stop); // in ms
    cudaEventElapsedTime(&timers.timer2, start2, stop2); // in ms

Error:
    cudaFree(arrayOnGpu);
    cudaFree(resultArrayFromGPU);

    return cudaStatus ? 1 : 0;
}