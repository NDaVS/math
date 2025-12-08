#include "my_serial.hpp"
#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <ctime>
#include <chrono>
#include <cstring>
#include <algorithm>

#if !defined (WIN32)
#	include <unistd.h>
#	include <time.h>
#endif

struct Measurement {
    std::time_t tm;
    double value;
};

static const std::string LOG_MEASURE = "measurements.log";
static const std::string LOG_HOURLY  = "hourly.log";
static const std::string LOG_DAILY   = "daily.log";


std::time_t now_ts() {
    return std::time(nullptr);
}

std::tm local_tm(std::time_t t) {
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    return tm;
}

// Сконвертировать любой базовый тип в строку
template<class T> std::string to_string(const T& v)
{
    std::ostringstream ss;
    ss << v;
    return ss.str();
}

bool parse_measurement(const std::string& s, double& outVal) {
    std::stringstream ss(s);
    double v;
    if (ss >> v) {
        outVal = v;
        return true;
    }
    return false;
}

void append_line(const std::string& file, const std::string& msg) {
    std::ofstream f(file, std::ios::app);
    f << msg << "\n";
}

void save_measurements(const std::string& file, const std::vector<Measurement>& v) {
    std::ofstream f(file, std::ios::trunc);
    for (auto& m : v) {
        f << std::to_string(m.tm) << " " << m.value << "\n";
    }
}



std::vector<Measurement> load_measurements(const std::string& file) {
    std::vector<Measurement> v;
    std::ifstream f(file);
    if (!f.is_open()) return v;

    std::string line;
    while (std::getline(f, line)) {
        std::stringstream ss(line);
        std::time_t t;
        double val;
        if (ss >> t >> val) {
            v.push_back({t, val});
        }
    }
    return v;
}

void trim_file_last_seconds(const std::string& file, int seconds) {
    auto v = load_measurements(file);
    std::time_t cutoff = now_ts() - seconds;
    std::vector<Measurement> out;

    for (auto& m : v) {
        if (m.tm >= cutoff)
            out.push_back(m);
    }
    save_measurements(file, out);
}

void trim_file_last_days(const std::string& file, int days) {
    auto v = load_measurements(file);
    std::time_t cutoff = now_ts() - days * 86400;
    std::vector<Measurement> out;

    for (auto& m : v) {
        if (m.tm >= cutoff)
            out.push_back(m);
    }
    save_measurements(file, out);
}

double compute_avg(const std::vector<double>& vals) {
    if (vals.empty()) return 0;
    double sum = 0;
    for (double v : vals) sum += v;
    return sum / vals.size();
}

void csleep(double timeout) {
#if defined (WIN32)
	if (timeout <= 0.0)
        ::Sleep(INFINITE);
    else
        ::Sleep((DWORD)(timeout * 1e3));
#else
    if (timeout <= 0.0)
        pause();
    else {
        struct timespec t;
        t.tv_sec = (int)timeout;
        t.tv_nsec = (int)((timeout - t.tv_sec)*1e9);
        nanosleep(&t, NULL);
    }
#endif
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cout << "Usage: prog_name [port]" << std::endl;
		return -1;
	}
	
	cplib::SerialPort smport(std::string(argv[1]),cplib::SerialPort::BAUDRATE_115200);
	if (!smport.IsOpen()) {
		std::cout << "Failed to open port '" << argv[1] << "'! Terminating..." << std::endl;
		return -2;
	}
	std::string mystr;
    smport.SetTimeout(1.0);

    std::vector<double> hour_vals;
    std::vector<double> day_vals;

    std::tm last_tm = local_tm(now_ts());
    int last_hour = last_tm.tm_min;
    int last_mday = last_tm.tm_mday;

    for (;;) {
        smport >> mystr;
        if (!mystr.empty()) {
            double temp;
            if (parse_measurement(mystr, temp)){
                std::time_t ts = now_ts();
                {
                    std::ofstream f(LOG_MEASURE, std::ios::app);
                    f << ts << " " << temp << "\n";
                }

                hour_vals.push_back(temp);
                day_vals.push_back(temp);
            }

            trim_file_last_seconds(LOG_MEASURE, 60);
        }

        std::tm cur_tm = local_tm(now_ts());
        if (cur_tm.tm_min != last_hour){
            double avg = compute_avg(hour_vals);
            append_line(LOG_HOURLY, std::to_string(now_ts()) + " " + std::to_string(avg));
            hour_vals.clear();
            last_hour = cur_tm.tm_min;

            trim_file_last_days(LOG_HOURLY, 30);
        }

        if (cur_tm.tm_mday != last_mday) {
            double avg = compute_avg(day_vals);
            append_line(LOG_DAILY, std::to_string(now_ts()) + " " + std::to_string(avg));

            day_vals.clear();
            last_mday = cur_tm.tm_mday;

            trim_file_last_days(LOG_DAILY, 365);

        }
    }

    return 0;
}
