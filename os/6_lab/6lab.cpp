#include "my_serial.hpp"
#include "sqlite3.h"
#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <ctime>
#include <chrono>
#include <iomanip>
#include <cstdio>
#include <thread>
#include <mutex>
#include <map>
#include <unordered_map>
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <unistd.h>
#include <time.h>
#endif

#ifdef _WIN32
#define socket_close closesocket
#else
#define socket_close close
#endif

std::mutex db_mutex;
double last_temperature = 0.0;
sqlite3 *db = nullptr;

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

void initialize_database()
{
    db = nullptr;

    if (sqlite3_open("test.db", &db) != SQLITE_OK)
    {
        std::cout << "open error\n";
    }

    const char *measurements =
        "CREATE TABLE IF NOT EXISTS measurements ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "value REAL NOT NULL,"
        "timestamp INTEGER NOT NULL"
        ");"

        "CREATE TABLE IF NOT EXISTS hourly_measurements ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "value REAL NOT NULL,"
        "timestamp INTEGER NOT NULL"
        ");"

        "CREATE TABLE IF NOT EXISTS daily_measurements ("
        "    id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "value REAL NOT NULL,"
        "timestamp INTEGER NOT NULL"
        ");";

    if (sqlite3_exec(db, measurements, nullptr, nullptr, nullptr) != SQLITE_OK)
    {
        std::cout << "sql error\n";
    }
}

void write_value(const std::string &table_name, double value, std::time_t timestamp)
{
    if (db == nullptr)
    {
        std::cerr << "Database not opened!" << std::endl;
        return;
    }

    std::string sql = "INSERT INTO " + table_name + " (value, timestamp) VALUES (?, ?);";

    sqlite3_stmt *stmt;
    int rc = sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK)
    {
        std::cerr << "SQL prepare error: " << sqlite3_errmsg(db) << std::endl;
        return;
    }

    sqlite3_bind_double(stmt, 1, value);
    sqlite3_bind_int64(stmt, 2, static_cast<sqlite3_int64>(timestamp));

    rc = sqlite3_step(stmt);
    if (rc != SQLITE_DONE)
    {
        std::cerr << "SQL step error: " << sqlite3_errmsg(db) << std::endl;
    }

    sqlite3_finalize(stmt);
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

std::string http_response(const std::string &body)
{
    std::ostringstream oss;
    oss << "HTTP/1.1 200 OK\r\n"
        << "Content-Type: application/json\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Connection: close\r\n\r\n"
        << body;
    return oss.str();
}

std::vector<std::pair<std::time_t, double>> query_by_time(const std::string &table, std::time_t from, std::time_t to)
{
    std::lock_guard<std::mutex> lock(db_mutex);

    std::string sql =
        "SELECT timestamp, value FROM " + table +
        " WHERE timestamp BETWEEN ? AND ?"
        " ORDER BY timestamp;";

    sqlite3_stmt *stmt;
    sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr);
    sqlite3_bind_int64(stmt, 1, from);
    sqlite3_bind_int64(stmt, 2, to);

    std::vector<std::pair<std::time_t, double>> result;

    while (sqlite3_step(stmt) == SQLITE_ROW)
    {
        std::time_t ts =
            static_cast<std::time_t>(sqlite3_column_int64(stmt, 0));

        double value =
            sqlite3_column_double(stmt, 1);

        result.emplace_back(ts, value);
    }

    sqlite3_finalize(stmt);
    return result;
}

bool net_init()
{
#ifdef _WIN32
    WSADATA wsa;
    return WSAStartup(MAKEWORD(2, 2), &wsa) == 0;
#else
    return true;
#endif
}

void net_cleanup()
{
#ifdef _WIN32
    WSACleanup();
#endif
}

std::string values_to_json(
    std::time_t from,
    std::time_t to,
    const std::vector<std::pair<std::time_t, double>> &values)
{
    std::ostringstream json;
    json.precision(10);

    json << "{";
    json << "\"from\":" << from << ",";
    json << "\"to\":" << to << ",";
    json << "\"count\":" << values.size() << ",";
    json << "\"values\":[";

    for (size_t i = 0; i < values.size(); ++i)
    {
        json << "{";
        json << "\"timestamp\":" << values[i].first << ",";
        json << "\"value\":" << values[i].second;
        json << "}";

        if (i + 1 < values.size())
            json << ",";
    }

    json << "]}";
    return json.str();
}


void http_server()
{
    if (!net_init())
        return;

#ifdef _WIN32
    SOCKET server_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
#else
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
#endif
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(8080);

    bind(server_fd, (sockaddr *)&addr, sizeof(addr));
    listen(server_fd, 5);
    std::cout << "HTTP server listening on port 8080\n";
    while (true)
    {
        int client = accept(server_fd, nullptr, nullptr);
        char buffer[1024]{};
        recv(client, buffer, sizeof(buffer) - 1, 0);
        std::string request(buffer);
        std::string body;

        auto line_end = request.find("\r\n");
        std::string request_line = request.substr(0, line_end);
        auto first_space = request.find(' ');
        auto second_space = request.find(' ', first_space + 1);

        std::string url = request_line.substr(
            first_space + 1,
            second_space - first_space - 1);

        std::string path;
        std::string query;

        auto qpos = url.find('?');
        if (qpos != std::string::npos)
        {
            path = url.substr(0, qpos);
            query = url.substr(qpos + 1);
        }
        else
        {
            path = url;
        }

        std::unordered_map<std::string, std::string> params;
        std::stringstream ss(query);
        std::string pair;

        while (std::getline(ss, pair, '&'))
        {
            auto eq = pair.find('=');
            if (eq != std::string::npos)
            {
                auto key = pair.substr(0, eq);
                auto value = pair.substr(eq + 1);
                params[key] = value;
            }
        }

        if (request.find("GET /current") != std::string::npos)
        {
            body = "{ \"temperature\": " + std::to_string(last_temperature) + " }";
        }
        else if (request.find("GET /stats") != std::string::npos)
        {
            std::time_t from = 0;
            std::time_t to = 0;
            if (params.count("from"))
                from = static_cast<std::time_t>(std::stoll(params["from"]));

            if (params.count("to"))
                to = static_cast<std::time_t>(std::stoll(params["to"]));
            
            std::cout << from << " " << to << std::endl;
            if (from == 0 || to == 0 || from >= to)
            {
                body = "{ \"error\": \"invalid time range\" }";
            }
            else
            {
                std::vector<std::pair<std::time_t, double>> values = query_by_time("measurements", from, to);
                body = values_to_json(from ,to, values);
            }
        }
        else
        {
            body = "{ \"error\": \"unknown endpoint\" }";
        }

        std::string resp = http_response(body);
        send(client, resp.c_str(), resp.size(), 0);
        socket_close(client);
    }
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
    initialize_database();
    std::thread(http_server).detach();

    smport.SetTimeout(1.0);

    std::vector<double> hour_vals;
    std::vector<double> day_vals;
    std::vector<double> measurement_buffer;

    std::tm last_tm = local_tm(now_ts());
    int last_hour = last_tm.tm_hour;
    int last_mday = last_tm.tm_mday;

    std::string line;
    std::cout << "Started!" << std::endl;
    for (;;)
    {
        smport >> line;
        if (!line.empty())
        {

            double value;
            std::string checksum;

            if (parse_packet(line, value, checksum))
            {

                if (value >= -60.0 && value <= 60.0 && (measurement_buffer.size() > 0 && measurement_buffer.back() - value < 1 || true))
                {
                    std::string calc = hash_of_string(std::to_string(value));
                    if (calc == checksum)
                    {
                        std::time_t ts = now_ts();

                        measurement_buffer.push_back(value);
                        last_temperature = value;
                        std::cout << "Received valid measurement: " << value << " at " << ts << std::endl;
                        write_value("measurements", value, ts);
                        if (measurement_buffer.size() >= 5)
                        {   // На случай если надо использовать буффер=)
                            // for (const auto &measurement : measurement_buffer)
                            // {
                            //     write_value("measurements", value, ts);
                            // }
                            measurement_buffer.clear();
                        }
                        hour_vals.push_back(value);
                        day_vals.push_back(value);
                    }
                }
            }
            std::time_t current_time = now_ts();
            std::tm cur_tm = local_tm(now_ts());

            if (cur_tm.tm_hour != last_hour)
            {
                double avg_hour = compute_avg(hour_vals);
                std::time_t hour_ts = current_time - (cur_tm.tm_min * 60 + cur_tm.tm_sec);
                write_value("hourly_measurements", avg_hour, hour_ts);
                hour_vals.clear();
                last_hour = cur_tm.tm_hour;
            }

            if (cur_tm.tm_mday != last_mday)
            {
                double avg_day = compute_avg(day_vals);
                std::tm day_start = cur_tm;
                day_start.tm_hour = 0;
                day_start.tm_min = 0;
                day_start.tm_sec = 0;
                std::time_t day_ts = std::mktime(&day_start);

                write_value("daily_measurements", avg_day, day_ts);
                day_vals.clear();
                last_mday = cur_tm.tm_mday;
            }
        }
    }

    sqlite3_close(db);
    return 0;
}