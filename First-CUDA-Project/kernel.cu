
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <algorithm>
#include <chrono>
#include <iostream>

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
constexpr char KEY_DEVICE_ID[] = "device-id";
constexpr char KEY_DURATION_TYPE[] = "duration-type";
constexpr char KEY_HEIGHT[] = "height";
constexpr char KEY_HELP[] = "help";
constexpr char KEY_KERNELS[] = "kernels";
constexpr char KEY_STREAMS[] = "streams";
constexpr char KEY_SHOW_DEVICES[] = "show-devices";
constexpr char KEY_SHOW_PARAMETERS[] = "show-parameters";
constexpr char KEY_OUTPUT[] = "output";
constexpr char KEY_TRANSFER[] = "transfer";
constexpr char KEY_TYPE[] = "type";
constexpr char KEY_WAIT_BEFORE_EXIT[] = "wait-before-exit";
constexpr char KEY_WIDTH[] = "width";

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
int checkCuda(cudaError_t result)
{
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        return 1;
    }

    return 0;
}

namespace gv
{
    class Timer {
    public:
        Timer() {};

        cudaEvent_t startH2D, stopH2D;
        cudaEvent_t startD2H, stopD2H;
        float timerH2D = -1.0f;
        float timerD2H = -1.0f;
    private:
        bool done{ false };
    };

    class Timers {
    public:
        Timers() {};
        ~Timers() {
            for (gv::Timer* ptr : listTimers_)
                delete ptr;
        };
        void pushBack(gv::Timer* timer) {
            listTimers_.push_back(timer);
        };
        void synchronizeAll() {
            for (auto* timer : listTimers_) {
                cudaEventElapsedTime(&timer->timerH2D, timer->startH2D, timer->stopH2D); // in ms
                cudaEventElapsedTime(&timer->timerD2H, timer->startD2H, timer->stopD2H); // in ms
            }
        };
        void printTimersToScreen() {
            for (auto* timer : listTimers_)
                std::cout << "H2D: " << timer->timerH2D << ", " << "D2H: " << timer->timerD2H << std::endl;
        }
        void printTimersToFile(const std::string& filename) {
            std::ofstream outputFile(filename);

            if (outputFile.is_open())
                for (auto* timer : listTimers_)
                    outputFile << "H2D: " << timer->timerH2D << ", " << "D2H: " << timer->timerD2H << std::endl;
            else
               std::cerr << "cannot open file!" << std::endl;


        }

    private:
        std::list <gv::Timer*> listTimers_;
    };

    Timers timers;

    class Parameters {
    public:
        Parameters() {};

        enum class DurationType {
            Invalid,
            Counter,
            Timer
        };

        enum class Transfer {
            Invalid,
            Pageable,
            Pinned
        };

        enum class NumericType {
            Invalid,
            Floating,
            Integer
        };

        const DurationType durationType(const std::string& key) {
            std::map<std::string, DurationType>::iterator it = mapOfStringDurationType.find(key);
            return (it != mapOfStringDurationType.end()) ? it->second : DurationType::Invalid;
        }

        const std::string string(const DurationType& value) {
            std::map<DurationType, std::string>::iterator it = mapOfDurationTypeString.find(value);
            return (it != mapOfDurationTypeString.end()) ? it->second : "invalid";
        }

        const Transfer transfer(const std::string& key) {
            std::map<std::string, Transfer>::iterator it = mapOfStringTransfer.find(key);
            return (it != mapOfStringTransfer.end()) ? it->second : Transfer::Invalid;
        }

        const std::string string(const Transfer& value) {
            std::map<Transfer, std::string>::iterator it = mapOfTransferString.find(value);
            return (it != mapOfTransferString.end()) ? it->second : "invalid";
        }

        const NumericType numericType(const std::string& key) {
            std::map<std::string, NumericType>::iterator it = mapOfStringNumericType.find(key);
            return (it != mapOfStringNumericType.end()) ? it->second : NumericType::Invalid;
        }

        const std::string string(const NumericType & value) {
            std::map<NumericType, std::string>::iterator it = mapOfNumericTypeString.find(value);
            return (it != mapOfNumericTypeString.end()) ? it->second : "invalid";
        }

        int validate()
        {
            int err = 0;

            if (channel_ < 1)
            {
                std::cerr << "Invalid channel option" << std::endl;
                err = 1;
            }
            else if (deviceId_ < 0)
            {
                std::cerr << "Invalid device-id option" << std::endl;
                err = 1;
            }
            else if (duration_ < 1)
            {
                std::cerr << "Invalid duration option" << std::endl;
                err = 1;
            }
            else if (durationType_ == DurationType::Invalid)
            {
                std::cerr << "Invalid duration type option" << std::endl;
                err = 1;
            }
            else if (height_ < 1)
            {
                std::cerr << "Invalid height option" << std::endl;
                err = 1;
            }
            else if (kernels_ < 1)
            {
                std::cerr << "Invalid kernel option" << std::endl;
                err = 1;
            }
            else if (streams_ < 1)
            {
                std::cerr << "Invalid streams option" << std::endl;
                err = 1;
            }
            else if (numericType_ == NumericType::Invalid)
            {
                std::cerr << "Invalid type option" << std::endl;
                err = 1;
            }
            else if (transfer_ == Transfer::Invalid)
            {
                std::cerr << "Invalid transfer option" << std::endl;
                err = 1;
            }
            else if (width_ < 1)
            {
                std::cerr << "Invalid width option" << std::endl;
                err = 1;
            }

            return err;
        }

        int applyOptions(cxxopts::ParseResult& result)
        {
            int err = 0;
            if (result.count(KEY_CONFIG))
            {
                std::string configFile = { result[KEY_CONFIG].as<std::string>() };
                err = readJsonFile(configFile);
            }

            if (err == 0)
            {
                if (result.count(KEY_ASYNC))
                    isAsync_ = result[KEY_ASYNC].as<bool>();

                if (result.count(KEY_CHANNEL))
                    channel_ = result[KEY_CHANNEL].as<int>();

                deviceId_ = result[KEY_DEVICE_ID].as<int>();

                if (result.count(KEY_DURATION))
                    duration_ = result[KEY_DURATION].as<int>();

                if (result.count(KEY_DURATION_TYPE))
                {
                    auto value = result[KEY_DURATION_TYPE].as<std::string>();
                    durationType_ = durationType(value);
                }

                if (result.count(KEY_HEIGHT))
                    height_ = result[KEY_HEIGHT].as<int>();

                kernels_ = result[KEY_KERNELS].as<int>();

                if (result.count(KEY_OUTPUT))
                {
                    auto output = result[KEY_OUTPUT].as<std::string>();
                    output_ = output;
                }

                if (result.count(KEY_STREAMS))
                {
                    auto streams = result[KEY_STREAMS].as<int>();
                    streams_ = streams;
                }

                if (result.count(KEY_TRANSFER))
                {
                    auto value = result[KEY_TRANSFER].as<std::string>();
                    transfer_ = transfer(value);
                }

                if (result.count(KEY_TYPE))
                {
                    auto value = result[KEY_TYPE].as<std::string>();
                    numericType_ = numericType(value);
                }

                if (result.count(KEY_WIDTH))
                    width_ = result[KEY_WIDTH].as<int>();
            }

            return err;
        }

        int getChannel() {
            return channel_;
        }

        int getDeviceId() {
            return deviceId_;
        }

        int getDuration() {
            return duration_;
        }

        DurationType getDurationType() {
            return durationType_;
        }

        int getHeight() {
            return height_;
        }

        int getKernels() {
            return kernels_;
        }

        std::string getOutput() {
            return output_;
        }

        int getStreams() {
            return streams_;
        }

        Transfer getTransfer() {
            return transfer_;
        }

        NumericType getNumeric() {
            return numericType_;
        }

        int getWidth() {
            return width_;
        }

        bool isAsync() {
            return isAsync_;
        }

        void printParameters()
        {
            std::cout << "isAsync: " << (isAsync_ ? "true" : "false") << std::endl;
            std::cout << "channel: " << channel_ << std::endl;
            std::cout << "deviceId" << deviceId_ << std::endl;
            std::cout << "duration: " << duration_ << std::endl;
            std::cout << "duration-type: " << string(durationType_) << std::endl;
            std::cout << "height: " << height_ << std::endl;
            std::cout << "kernels: " << kernels_ << std::endl;
            std::cout << "output: " << output_ << std::endl;
            std::cout << "streams: " << streams_ << std::endl;
            std::cout << "transfer: " << string(transfer_) << std::endl;
            std::cout << "numeric type: " << string(numericType_) << std::endl;
            std::cout << "width: " << width_ << std::endl;
        }

    private:
        int readJsonFile(const std::string configFile)
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

            if (doc.HasMember(KEY_ASYNC) && doc[KEY_ASYNC].IsBool())
                isAsync_ = doc[KEY_ASYNC].GetBool();

            if (doc.HasMember(KEY_CHANNEL) && doc[KEY_CHANNEL].IsInt())
                channel_ = doc[KEY_CHANNEL].GetInt();

            if (doc.HasMember(KEY_DEVICE_ID) && doc[KEY_DEVICE_ID].IsInt())
                deviceId_ = doc[KEY_DEVICE_ID].GetInt();

            if (doc.HasMember(KEY_DURATION) && doc[KEY_DURATION].IsInt())
                duration_ = doc[KEY_DURATION].GetInt();

            if (doc.HasMember(KEY_DURATION_TYPE) && doc[KEY_DURATION_TYPE].IsString())
            {
                auto value = doc[KEY_DURATION_TYPE].GetString();
                durationType_ = durationType(value);
            }

            if (doc.HasMember(KEY_HEIGHT) && doc[KEY_HEIGHT].IsInt())
                height_ = doc[KEY_HEIGHT].GetInt();

            if (doc.HasMember(KEY_KERNELS) && doc[KEY_KERNELS].IsInt())
                kernels_ = doc[KEY_KERNELS].GetInt();

            if (doc.HasMember(KEY_STREAMS) && doc[KEY_STREAMS].IsInt())
                streams_ = doc[KEY_STREAMS].GetInt();

            if (doc.HasMember(KEY_TRANSFER) && doc[KEY_TRANSFER].IsString())
            {
                auto value = doc[KEY_TRANSFER].GetString();
                transfer_ = transfer(value);
            }

            if (doc.HasMember(KEY_TYPE) && doc[KEY_TYPE].IsString())
            {
                auto value = doc[KEY_TYPE].GetString();
                numericType_ = numericType(std::string(value));
            }

            if (doc.HasMember(KEY_OUTPUT) && doc[KEY_OUTPUT].IsString())
            {
                auto output = doc[KEY_OUTPUT].GetString();
                output_ = doc[KEY_OUTPUT].GetString();
            }

            if (doc.HasMember(KEY_WIDTH) && doc[KEY_WIDTH].IsInt())
                width_ = doc[KEY_WIDTH].GetInt();

            return 0;
        }

        // types
        std::map<std::string, DurationType> mapOfStringDurationType = {
            {"counter", DurationType::Counter},
            {"timer", DurationType::Timer}
        };
        std::map<DurationType, std::string> mapOfDurationTypeString = {
            {DurationType::Counter, "counter"},
            {DurationType::Timer, "timer"}
        };
        std::map<std::string, Transfer> mapOfStringTransfer = {
            {"pageable", Transfer::Pageable},
            {"pinned", Transfer::Pinned}
        };
        std::map<Transfer, std::string> mapOfTransferString = {
            {Transfer::Pageable, "pageable"},
            {Transfer::Pinned, "pinned"}
        };
        std::map<std::string, NumericType> mapOfStringNumericType = {
            {"floating", NumericType::Floating},
            {"integer", NumericType::Integer}
        };
        std::map<NumericType, std::string> mapOfNumericTypeString = {
            {NumericType::Floating, "floating"},
            {NumericType::Integer, "integer"}
        };

        // parameters
        int channel_{ -1 };
        int deviceId_{ -1 };
        int duration_{ -1 };
        DurationType durationType_{ DurationType::Invalid };
        int height_{ -1 };
        int kernels_{ -1 };
        std::string output_{};
        int streams_{ -1 };
        Transfer transfer_{ Transfer::Invalid };
        NumericType numericType_{ NumericType::Invalid };
        int width_{ -1 };
        bool isAsync_{ false };
    };

    namespace cuda
    {
        template<class T>
        __global__ void addKernel(T* dev, int size)
        {
            for (int i = 0; i < size; i++)
                dev[i] += 1;
        }

        void showDevices() {
            int count = -1;

            cudaGetDeviceCount(&count);
            std::cout << "Number of available devices: " << count << std::endl;

            for (int i = 0; i < count; ++i) {
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, i);
                std::cout << i << ":" << prop.name << std::endl;
            }

        }
    }

    namespace app
    {
        cxxopts::ParseResult result;
        cxxopts::Options options("first-CUDA-Project", "copy back and forth from host to GPU an array and measure times.");

        int processCmdLineOpts(int argc, char** argv)
        {
            int err = 0;

            options.add_options()
                (KEY_ASYNC, "Copy using streams (2).", cxxopts::value<bool>()->default_value("false"))
                (KEY_CHANNEL, "Number of channels for the array.", cxxopts::value<int>())
                (KEY_CONFIG, "Config file to load.", cxxopts::value<std::string>())
                (KEY_DEVICE_ID, "Device ID to use.", cxxopts::value<int>()->default_value("0"))
                (KEY_DURATION, "Number of iterations or time in seconds.", cxxopts::value<int>())
                (KEY_DURATION_TYPE, "Possible value is counter or timer.", cxxopts::value<std::string>())
                (KEY_HEIGHT, "Height of the array.", cxxopts::value<int>())
                (KEY_KERNELS, "Number of kernels to apply.", cxxopts::value<int>()->default_value("1"))
                (KEY_OUTPUT, "Store to file instead of print to screen.", cxxopts::value<std::string>())
                (KEY_SHOW_DEVICES, "Show devices available and exit.")
                (KEY_SHOW_PARAMETERS, "Show parameters (if validation succeded) and exit.")
                (KEY_STREAMS, "Number of streams.", cxxopts::value<int>()->default_value("1"))
                (KEY_TRANSFER, "Transfer memory mode. possible options are \"pageable\" and \"pinned\".", cxxopts::value<std::string>()->default_value("pageable"))
                (KEY_WAIT_BEFORE_EXIT, "After process finished, it will be asked to press \"enter\" before returning to the command line.")
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
                err = 1;
            }

            return err;
        }

        void pause() {
            std::cout << "Press Enter to Continue";
#undef max // for visual studio intellisense...
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }

        template <class T>
        int copyCUDA(gv::Parameters& parameters, T* dst, T* src, enum cudaMemcpyKind cpyKind, cudaStream_t& stream)
        {
            const int height = parameters.getHeight();
            const int width = parameters.getWidth();
            const int channels = parameters.getChannel();
            const int size = height * width * channels;

            int err = 0;

            if (parameters.isAsync())
            {

                err = checkCuda(cudaMemcpyAsync(dst, src, size * sizeof(T), cpyKind, stream));
            }
            else
            {
                err = checkCuda(cudaMemcpy(dst, src, size * sizeof(T), cpyKind));
            }

            return err;
        }

        template <class T>
        int copy(gv::Parameters& parameters, T* host, T* D2H, gv::Timer* timer, cudaStream_t& stream)
        {
            int err = 0;
            T* dev = 0;
            nvtxEventAttributes_t eventAttrib = { 0 }; // zero the structure
            const int deviceId = parameters.getDeviceId();
            const int height = parameters.getHeight();
            const int kernels = parameters.getKernels();
            const int width = parameters.getWidth();
            const int channels = parameters.getChannel();
            const int size = height * width * channels;
            const uint32_t COLOR_GREEN = 0xFF00FF00;

            if (checkCuda(cudaEventCreate(&timer->startH2D)))
                err = 1;
            else if (checkCuda(cudaEventCreate(&timer->stopH2D)))
                err = 1;
            else if (checkCuda(cudaEventCreate(&timer->startD2H)))
                err = 1;
            else if (checkCuda(cudaEventCreate(&timer->stopD2H)))
                err = 1;
            else if (checkCuda(cudaSetDevice(deviceId))) // Choose which GPU to run on
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

                if (checkCuda(cudaEventRecord(timer->startH2D, stream)))
                    err = 1;

                if (copyCUDA<T>(parameters, dev, host, cudaMemcpyHostToDevice, stream))
                    err = 1;

                if (checkCuda(cudaEventRecord(timer->stopH2D, stream)))
                    err = 1;

                nvtxRangePop();

                // <Launch a kernel on the GPU here>
                for (int i = 0; i < kernels; i++)
                    gv::cuda::addKernel<T> << <1, 1, 0, stream >> > (dev, size); // using thread, keep in mind to don't go over max cores.
            }


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
                if (checkCuda(cudaEventRecord(timer->startD2H, stream)))
                    err = 1;

                err = copyCUDA(parameters, D2H, dev, cudaMemcpyDeviceToHost, stream);

                if (checkCuda(cudaEventRecord(timer->stopD2H, stream)))
                    err = 1;
                nvtxRangePop();

            }

            // XXX est-ce que les sync sont necessaires ?
            // Synchronize CUDA events and get timers
            if (!err)
            {
                //cudaEventSynchronize(stopH2D); //put that in a thread ??
                //cudaEventSynchronize(stopD2H);

                if (checkCuda(cudaEventSynchronize(timer->startH2D)))
                {
                    std::cout << "X11" << std::endl;
                    err = 1;
                }

                if (checkCuda(cudaEventSynchronize(timer->startD2H)))
                {
                    std::cout << "X12" << std::endl;
                    err = 1;
                }

                if (checkCuda(cudaEventSynchronize(timer->stopH2D)))
                {
                    std::cout << "X13" << std::endl;
                    err = 1;
                }

                if (checkCuda(cudaEventSynchronize(timer->stopD2H)))
                {
                    std::cout << "X14" << std::endl;
                    err = 1;
                }

            }

            return err;
        }

        template <class T>
        int startCounterLoop(gv::Parameters& parameters, T* host, T* D2H)
        {
            const int duration = parameters.getDuration();
            const int streams = parameters.getStreams();
            int err = 0;

            std::list<cudaStream_t*> streamList{};

            for (int i = 0; i < streams; i++)
            {
                auto stream = new cudaStream_t();
                cudaStreamCreate(stream);
                streamList.push_back(stream);
            }

            auto it = streamList.begin();
            for (int i = 0; i < duration; i++)
            {
                auto timer = new gv::Timer();

                if (err = gv::app::copy(parameters, host, D2H, timer, *(*it)), err)
                    break;

                gv::timers.pushBack(timer);

                if (++it == streamList.end())
                    it = streamList.begin();
            }

            for (auto* stream : streamList)
                cudaStreamSynchronize(*stream);

            gv::timers.synchronizeAll();

            for (auto* stream : streamList)
                delete stream;

            return err;
        }

        template <class T>
        int startTimerLoop(gv::Parameters& parameters, T* host, T* D2H)
        {
            const int duration = parameters.getDuration();
            const int streams = parameters.getStreams();
            int err = 0;

            std::list<cudaStream_t> streamList{};

            for (int i = 0; i < streams; i++)
            {
                streamList.push_back({});
                cudaStream_t stream = *streamList.end();
                cudaStreamCreate(&stream);
            };

            auto start = std::chrono::system_clock::now();
            auto end = std::chrono::system_clock::now();
            auto it = streamList.begin();
            while ((std::chrono::duration_cast<std::chrono::seconds>(end - start).count() != duration))
            {
                auto timer = new gv::Timer();

                if (err = gv::app::copy(parameters, host, D2H, timer, *it), err)
                    break;

                gv::timers.pushBack(timer);

                if (++it == streamList.end())
                    it = streamList.begin();

                end = std::chrono::system_clock::now();
            }

            return err;
        }

        template <class T>
        int startCopyTest(gv::Parameters& parameters)
        {
            const int height = parameters.getHeight();
            const int width = parameters.getWidth();
            const int channels = parameters.getChannel();
            const int size = height * width * channels;
            int err = 0;

            T* host = nullptr;
            T* D2H = nullptr;

            // arrays allocation
            switch (parameters.getTransfer())
            {
            case gv::Parameters::Transfer::Pageable:
            {
                host = (T*)malloc(sizeof(T) * size);
                D2H = (T*)malloc(sizeof(T) * size);
                break;
            }
            case gv::Parameters::Transfer::Pinned:
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
            case::gv::Parameters::Transfer::Invalid:
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
                // fill arrays
                for (int i = 0; i < size; ++i)
                {
                    host[i] = static_cast<T>(1.0);
                    D2H[i] = static_cast <T>(0.0);
                }

                if (parameters.getDurationType() == gv::Parameters::DurationType::Counter && !err)
                    err = gv::app::startCounterLoop<T>(parameters, host, D2H);
                else if (parameters.getDurationType() == gv::Parameters::DurationType::Timer && !err)
                    err = gv::app::startTimerLoop<T>(parameters, host, D2H);

                // free allocations
                switch (parameters.getTransfer())
                {
                case gv::Parameters::Transfer::Pageable:
                {
                    if (host)
                        free(host);
                    if (D2H)
                        free(D2H);
                    break;
                }
                case gv::Parameters::Transfer::Pinned:
                {
                    if (host)
                        cudaFreeHost(host);
                    if (D2H)
                        cudaFreeHost(D2H);
                    break;
                }
                case::gv::Parameters::Transfer::Invalid:
                    err = 1;
                }
            }

            return err;
        }

        int process(gv::Parameters& parameters)
        {
            int err = 0;
            auto numericType = parameters.getNumeric();

            switch (numericType)
            {
            case gv::Parameters::NumericType::Floating:
                err = gv::app::startCopyTest<float>(parameters);
                break;
            case gv::Parameters::NumericType::Integer:
                err = gv::app::startCopyTest<int>(parameters);
                break;
            case gv::Parameters::NumericType::Invalid:
                std::cerr << "Invalid type!" << std::endl;
                err = 1;
                break;
            }

            // cudaDeviceReset must be called before exiting in order for profiling and
            // tracing tools such as Nsight and Visual Profiler to show complete traces.

            err = checkCuda(cudaDeviceReset());

            return err;
        }
    }
}

int main(int argc, char** argv)
{
    gv::Parameters parameters;
    int err = 0;

    if (gv::app::processCmdLineOpts(argc, argv))
        std::cerr << "(EE) gv::app::cxxopts -> error" << std::endl;
    else if (gv::app::result.count(KEY_HELP))
        std::cout << gv::app::options.help() << std::endl;
    else if (gv::app::result.count(KEY_SHOW_DEVICES))
        gv::cuda::showDevices();
    else if (err = parameters.applyOptions(gv::app::result), err)
        std::cerr << "(EE) applyOptions -> error" << std::endl;
    else if (err = parameters.validate(), err)
        std::cerr << "(EE) validate -> error" << std::endl;
    else if (gv::app::result.count(KEY_SHOW_PARAMETERS))
        parameters.printParameters();
    else if (err = gv::app::process(parameters), err)
        std::cerr << "(EE) process -> error" << std::endl;
    else if (gv::app::result.count(KEY_OUTPUT))
        gv::timers.printTimersToFile(parameters.getOutput());
    else
        gv::timers.printTimersToScreen();

    if (gv::app::result.count(KEY_WAIT_BEFORE_EXIT))
        gv::app::pause();

   return err;
}