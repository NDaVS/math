#include "my_serial.hpp"

#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <ctime>
#include <chrono>
#include <iomanip>
#include <cstdio>

#if !defined(WIN32)
#include <unistd.h>
#include <time.h>
#endif

struct Measurement
{
    std::time_t tm;
    double value;
};

static const std::string LOG_MEASURE = "measurements.log";
static const std::string LOG_HOURLY = "hourly.log";
static const std::string LOG_DAILY = "daily.log";

static const int max_measurements_log = 24 * 60 * 60;
static const int max_hourly_log = 24 * 30;
static const int max_daily_log = 365;

std::time_t now_ts()
{
    return std::time(nullptr);
}

std::tm local_tm(std::time_t t)
{
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    return tm;
}

double compute_avg(const std::vector<double> &vals)
{
    if (vals.empty())
        return 0.0;
    double sum = 0.0;
    for (double v : vals)
        sum += v;
    return sum / vals.size();
}

std::size_t count_lines(const std::string &file)
{
    std::ifstream f(file);
    if (!f.is_open())
        return 0;

    std::size_t count = 0;
    std::string line;
    while (std::getline(f, line))
        ++count;

    return count;
}
void reset_if_oversize(const std::string &file, int type)
{
    int limit = 0;

    switch (type)
    {
    case 0:
        limit = max_measurements_log;
        break;
    case 1:
        limit = max_hourly_log;
        break;
    case 2:
        limit = max_daily_log;
        break;
    default:
        return;
    }

    std::size_t lines = count_lines(file);

    if (lines >= static_cast<std::size_t>(limit))
    {
        std::remove(file.c_str());
    }
}

void append_line(const std::string &file,
                 const std::string &msg,
                 int type)
{
    reset_if_oversize(file, type);

    std::ofstream f(file, std::ios::app);
    f << msg << "\n";
}

std::string hash_of_string(const std::string &s)
{
    std::stringstream hex_stream;

    for (char c : s)
    {
        hex_stream << std::hex << std::setw(2) << std::setfill('0')
                   << static_cast<int>(static_cast<unsigned char>(c));
    }

    std::string result = hex_stream.str();
    for (char &c : result)
    {
        c = std::toupper(static_cast<unsigned char>(c));
    }

    return result;
}

bool parse_packet(const std::string &s, double &value, std::string &checksum)
{
    std::stringstream ss(s);

    ss >> value >> checksum;

    return !ss.fail();
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: [progname] [port]\n";
        return -1;
    }

    cplib::SerialPort smport(argv[1], cplib::SerialPort::BAUDRATE_115200);
    if (!smport.IsOpen())
    {
        std::cout << "Failed to open port\n";
        return -2;
    }

    smport.SetTimeout(1.0);

    std::vector<double> hour_vals;
    std::vector<double> day_vals;
    std::vector<std::pair<std::time_t, double>> measurement_buffer;

    std::tm last_tm = local_tm(now_ts());
    int last_hour = last_tm.tm_hour;
    int last_mday = last_tm.tm_mday;

    std::string line;

    for (;;)
    {
        smport >> line;
        if (!line.empty())
        {
            double value;
            std::string checksum;

            if (parse_packet(line, value, checksum))
            {
                bool value_ok = (value >= -60.0 && value <= 60.0);

                if (!measurement_buffer.empty())
                {
                    value_ok = value_ok &&
                               (std::abs(value - measurement_buffer.back().second) < 1.0);
                }

                if (value_ok)
                {
                    std::string calc = hash_of_string(std::to_string(value));
                    if (calc == checksum)
                    {
                        std::cout << "New value -> " << value << std::endl;
                        std::time_t ts = now_ts();

                        measurement_buffer.emplace_back(ts, value);

                        if (measurement_buffer.size() >= 10)
                        {
                            for (const auto &measurement : measurement_buffer)
                            {
                                append_line(
                                    LOG_MEASURE,
                                    std::to_string(measurement.first) + " " +
                                        std::to_string(measurement.second),
                                    0);
                            }
                            measurement_buffer.clear();

                            hour_vals.push_back(value);
                            day_vals.push_back(value);
                        }
                    }
                }
            }

            std::tm cur_tm = local_tm(now_ts());

            if (cur_tm.tm_hour != last_hour)
            {
                append_line(
                    LOG_HOURLY,
                    std::to_string(now_ts()) + " " +
                        std::to_string(compute_avg(hour_vals)),
                    1);

                hour_vals.clear();
                last_hour = cur_tm.tm_hour;
            }

            if (cur_tm.tm_mday != last_mday)
            {
                append_line(
                    LOG_DAILY,
                    std::to_string(now_ts()) + " " +
                        std::to_string(compute_avg(day_vals)),
                    2);
            }
        }
    }
}