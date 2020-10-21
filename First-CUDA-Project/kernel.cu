
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
#include <Windows.h>


constexpr char KEY_ASYNC[] = "async";
constexpr char KEY_CHANNEL[] = "channel";
constexpr char KEY_CONFIG[] = "config";
constexpr char KEY_DURATION[] = "duration";
constexpr char KEY_DURATION_TYPE[] = "duration-type";
constexpr char KEY_HEIGHT[] = "height";
constexpr char KEY_HELP[] = "help";
constexpr char KEY_OUTPUT[] = "output";
constexpr char KEY_TRANSFER[] = "transfer";
constexpr char KEY_TYPE[] = "type";
constexpr char KEY_WIDTH[] = "width";

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
int checkCuda(cudaError_t result)
{
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        return 1;
    }

    return 0;
}

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


enum class Transfer {
    Invalid,
    Pageable,
    Pinned
};

Transfer stringToTransfer(const std::string& key)
{
    Transfer result = Transfer::Invalid;

    std::map<std::string, Transfer> mapOfType = {
        {"pageable", Transfer::Pageable},
        {"pinned", Transfer::Pinned}
    };

    std::map<std::string, Transfer>::iterator it = mapOfType.find(key);

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
    std::string output{};
    Transfer transfer = Transfer::Invalid;
    Type type = Type::Invalid;
    int width = -1;
    bool isAsync{ false };
};

struct Timers {
    float timerH2D = -1.0f;
    float timerD2H = -1.0f;
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

    if(doc.HasMember(KEY_ASYNC) && doc[KEY_ASYNC].IsBool())
        parameters.isAsync = doc[KEY_ASYNC].GetBool();

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

    if (doc.HasMember(KEY_TRANSFER) && doc[KEY_TRANSFER].IsString())
    {
        auto type = doc[KEY_TRANSFER].GetString();
        parameters.transfer = application::stringToTransfer(type);
    }

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
        std::cerr << "Invalid channel option" << std::endl;
        err = 1;
    }
    else if (parameters.duration < 1)
    {
        std::cerr << "Invalid duration option" << std::endl;
        err = 1;
    } 
    else if (parameters.durationType == application::DurationType::Invalid)
    {
        std::cerr << "Invalid duration type option" << std::endl;
        err = 1;
    }
    else if (parameters.height < 1)
    {
        std::cerr << "Invalid height option" << std::endl;
        err = 1;
    }
    else if (parameters.type == application::Type::Invalid)
    {
        std::cerr << "Invalid type option" << std::endl;
        err = 1;
    }
    else if (parameters.transfer == application::Transfer::Invalid)
    {
        std::cerr << "Invalid transfer option" << std::endl;
        err = 1;
    }
    else if (parameters.width< 1)
    {
        std::cerr << "Invalid width option" << std::endl;
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
        if (result.count(KEY_ASYNC))
            parameters.isAsync = result[KEY_ASYNC].as<bool>();

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

        if (result.count(KEY_TRANSFER))
        {
            auto transfer = result[KEY_TRANSFER].as<std::string>();
            parameters.transfer = application::stringToTransfer(transfer);
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
int copy(const application::Parameters& parameters, T* host, T* D2H, application::Timers &timers);

template <class T>
int startCounterLoop(const application::Parameters& parameters, T* host, T* D2H, std::list<application::Timers> &listOfTimers)
{
    const int duration = parameters.duration;
    int err = 0;

    for (int i = 0; i < duration; i++)
    {
        application::Timers timers;
        if (err = copy(parameters, host, D2H, timers), err)
            break;
        listOfTimers.push_back(timers);
    }

    return err;
}

template <class T>
int startTimerLoop(application::Parameters& parameters, T* host, T* D2H, std::list<application::Timers> & listOfTimers)
{
    const int duration = parameters.duration;
    int err = 0;

    auto start = std::chrono::system_clock::now();
    auto end = std::chrono::system_clock::now();
    while ((std::chrono::duration_cast<std::chrono::seconds>(end - start).count() != duration))
    {
        application::Timers timers;
        if (err = copy(parameters, host, D2H, timers), err)
            break;
        listOfTimers.push_back(timers);
        end = std::chrono::system_clock::now();
    }

    return err;
}

void printToScreen(std::list<application::Timers> &listOfTimers)
{
    for (auto const& i : listOfTimers)
        std::cout << i.timerH2D << ", " << i.timerD2H << std::endl;
}

void printToFile(const std::string &output, std::list<application::Timers>& listOfTimers)
{
    std::ofstream outputFile(output);

    if (outputFile.is_open())
        for (auto const& i : listOfTimers)
            outputFile << i.timerH2D << ", " << i.timerD2H << std::endl;
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
void fillArrays(application::Parameters& parameters, T* host, T* D2H)
{
    const int height = parameters.height;
    const int width = parameters.width;
    const int channels = parameters.channel;
    const int size = height * width * channels;

    for (int i = 0; i < size; ++i)
    {
        host[i] = static_cast<T>(1.0);
        D2H[i] = static_cast <T>(0.0);
    }
}

template <class T>
int startCopyTest(application::Parameters &parameters, std::list<application::Timers> &listOfTimers)
{
    const int height = parameters.height;
    const int width = parameters.width;
    const int channels = parameters.channel; 
    const int size = height * width * channels;
    int err = 0;

    T* host = nullptr;
    T* D2H = nullptr;
    
    // arrays allocation
    switch (parameters.transfer)
    {
        case application::Transfer::Pageable:
        {
            host = (T*)malloc(sizeof(T) * size);
            D2H = (T*)malloc(sizeof(T) * size);
            break;
        }
        case application::Transfer::Pinned:
        {
            if (err = checkCuda(cudaMallocHost((T**)&host, size * sizeof(T))), err)
            {
                break;
            }
            else if (err = checkCuda(cudaMallocHost((T**)&D2H, size * sizeof(T))), err)
            {
                break;
            }
            break;
        }
        case::application::Transfer::Invalid:
            err = 1;
    }
            

    if (host == nullptr)
    {
        std::cerr << "host allocation failed!" << std::endl;
        err = 1;
    }
    else if (D2H == nullptr)
    {
        std::cerr << "D2H allocation failed!" << std::endl;
        err = 1;
    }

    if (!err)
    {
        fillArrays(parameters, host, D2H);

        if (parameters.durationType == application::DurationType::Counter && !err)
            err = startCounterLoop<T>(parameters, host, D2H, listOfTimers);
        else if (parameters.durationType == application::DurationType::Timer && !err)
            err = startTimerLoop<T>(parameters, host, D2H, listOfTimers);

        // free allocations
        switch (parameters.transfer)
        {
        case application::Transfer::Pageable:
        {
            if (host)
                free(host);
            if (D2H)
                free(D2H);
            break;
        }
        case application::Transfer::Pinned:
        {
            if (host)
                cudaFreeHost(host);
            if (D2H)
                cudaFreeHost(D2H);
            break;
        }
        case::application::Transfer::Invalid:
            err = 1;
        }
    }
    
    
    return err;
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
    // enum to string are limited.
    std::cout << "isAsync: " << (parameters.isAsync ? "true" : "false") << std::endl;
    std::cout << "channel: " << parameters.channel << std::endl;
    std::cout << "duration: " << parameters.duration << std::endl;
    std::cout << "duration-type: " << (parameters.durationType == application::DurationType::Timer ? "Timer" : "Counter") << std::endl;
    std::cout << "height: " << parameters.height << std::endl;
    std::cout << "output: " << parameters.output << std::endl;
    std::cout << "transfer: " << (parameters.transfer  == application::Transfer::Pageable ? "Pageable" : "Pinned")<< std::endl;
    std::cout << "type: " << (parameters.type == application::Type::Floating ? "Floating" : "Integer") << std::endl;
    std::cout << "width: " << parameters.width << std::endl;
}

int main(int argc, char** argv)
{
    application::Parameters parameters;
    std::list<application::Timers> listOfTimers;
    int err = 0;

    cxxopts::ParseResult result;
    cxxopts::Options options("first-CUDA-Project", "copy back and forth from host to GPU an array and measure times.");

    options.add_options()
        (KEY_ASYNC, "Copy using streams (2).", cxxopts::value<bool>()->default_value("false"))
        (KEY_CHANNEL, "Number of channels for the array.", cxxopts::value<int>())
        (KEY_CONFIG, "Config file to load.", cxxopts::value<std::string>())
        (KEY_DURATION, "Number of iterations or time in seconds.", cxxopts::value<int>())
        (KEY_DURATION_TYPE, "Possible value is counter or timer.", cxxopts::value<std::string>())
        (KEY_HEIGHT, "Height of the array.", cxxopts::value<int>())
        (KEY_OUTPUT, "Store to file instead of print to screen.", cxxopts::value<std::string>())
        (KEY_TRANSFER, "Transfer memory mode. possible options are \"pageable\" and \"pinned\".", cxxopts::value<std::string>()->default_value("pageable"))
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
        std::cerr << "(EE) startCopyTest2 -> error" << std::endl;
    else
    {
        print(parameters, listOfTimers);
        //print(parameters);
    }
    
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.

    err = checkCuda(cudaDeviceReset());
    /*
    auto cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        std::cerr <<  "cudaDeviceReset failed!" << std::endl;
        err = 1;
    }
    */

   // std::cout << "Press Enter to ContinueX";
//#undef max // for visual studio intellisense...
  //  std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

   return err;
}

template <class T>
int copyCUDA(const application::Parameters& parameters, T* dst, T* src, enum cudaMemcpyKind cpyKind)
{
    const int height = parameters.height;
    const int width = parameters.width;
    const int channels = parameters.channel;
    const int size = height * width * channels;

    int err = 0;

    if (parameters.isAsync)
    {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        err = checkCuda(cudaMemcpyAsync(dst, src, size * sizeof(T), cpyKind, stream));
    }
    else
    {
        err = checkCuda(cudaMemcpy(dst, src, size * sizeof(T), cpyKind));
    }

    return err;
}

template <class T>
int copy(const application::Parameters& parameters, T* host, T* D2H, application::Timers& timers)
{

    const int height = parameters.height;
    const int width = parameters.width;
    const int channels = parameters.channel;
    const int size = height * width * channels;
    const uint32_t COLOR_GREEN = 0xFF00FF00;

    //cudaError_t cudaStatus;
    cudaEvent_t startH2D, stopH2D;
    cudaEventCreate(&startH2D);
    cudaEventCreate(&stopH2D);
    cudaEvent_t startD2H, stopD2H;
    cudaEventCreate(&startD2H);
    cudaEventCreate(&stopD2H);

    T* dev = 0;
    int err = 0;
    nvtxEventAttributes_t eventAttrib = { 0 }; // zero the structure

    // Choose which GPU to run on, change this on a multi-GPU system.
    if (checkCuda(cudaSetDevice(0)))
        err = 1;
    else if (checkCuda(cudaMalloc((void**)&dev, size * sizeof(T))))
        err = 1;
    else
    {
        // Copy input vectors from host memory to GPU buffers.
        eventAttrib = { 0 }; // zero the structure
        eventAttrib.version = NVTX_VERSION;
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        eventAttrib.colorType = NVTX_COLOR_ARGB;
        eventAttrib.color = COLOR_GREEN;
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
        eventAttrib.message.ascii = __FUNCTION__ ":timerH2D ";

        nvtxRangePushEx(&eventAttrib);
        cudaEventRecord(startH2D);

        //err = checkCuda(cudaMemcpy(dev, host, size * sizeof(T), cudaMemcpyHostToDevice))
        err = copyCUDA<T>(parameters, dev, host, cudaMemcpyHostToDevice);

        cudaEventRecord(stopH2D);
        nvtxRangePop();

    }

    // <Launch a kernel on the GPU here>

    // Copy output vector from GPU buffer to host memory.
    if (!err)
    {
        eventAttrib = { 0 }; // zero the structure
        eventAttrib.version = NVTX_VERSION;
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        eventAttrib.colorType = NVTX_COLOR_ARGB;
        eventAttrib.color = COLOR_GREEN;
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
        eventAttrib.message.ascii = __FUNCTION__ ":timerD2H ";

        nvtxRangePushEx(&eventAttrib);
        cudaEventRecord(startD2H);
        //err = checkCuda(cudaMemcpy(D2H, dev, size * sizeof(T), cudaMemcpyDeviceToHost))
        err = copyCUDA(parameters, D2H, dev, cudaMemcpyDeviceToHost);

        cudaEventRecord(stopD2H);
        nvtxRangePop();

    }

    // Synchronize CUDA events and get timers
    if (!err)
    {
        cudaEventSynchronize(stopH2D);
        cudaEventSynchronize(stopD2H);

        cudaEventElapsedTime(&timers.timerH2D, startH2D, stopH2D); // in ms
        cudaEventElapsedTime(&timers.timerD2H, startD2H, stopD2H); // in ms
    }

    return err;
}