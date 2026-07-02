/**
 * Copyright (c) 2025 QINGMAO INTELLIGENCE TECHNOLOGY (BEIJING) CO., LTD. and Huawei Technologies Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "string_utils.h"
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>

class LogStream
{
    std::ostream& stream_;

public:
    explicit LogStream(std::ostream& stream)
        : stream_(stream)
    {
    }

    ~LogStream()
    {
        stream_ << std::endl;
    }

    template <typename T>
    LogStream& operator<<(const T& value)
    {
        stream_ << value;
        return *this;
    }
};

class Logger
{
public:
    enum Level
    {
        TRACE = 0,
        DEBUG = 10,
        INFO = 20,
        WARNING = 30,
        ERROR = 40
    };

    static Logger& GetInstance();
    ~Logger();
    Logger(const Logger&) = delete;
    void operator=(const Logger&) = delete;

    void SetLogFilePath(const std::string& dir);
    const std::string& GetOutputDir() const;
    void SetRank(int rank);
    int GetRank() const;
    void log(std::exception& ex, Level level = Level::ERROR);
    Level GetLevel() const;
    void SetLevel(const Level level);
    bool IsEnabled(const Level level) const;

    template <typename... Args>
    void log(Level level, const char* file, int line, const char* format, const Args&... args)
        __attribute__((format(printf, 5, 0)));

    template <typename... Args>
    void log(Level level, const char* file, int line, const std::string& format, const Args&... args);

private:
    std::string outputDir;
    std::ofstream outFile;
    int rank_{0};
    std::mutex logMutex;
    Level level_ = Level::INFO;

    Logger();
    static const char* GetLevelName(const Level level);
    static std::string GetPrefix(const Level level);
    static std::string GetPrefix(const Level level, const int rank);
    static std::string GetPrefix(const Level level, const char* file, int line);
    static std::string GetPrefix(const Level level, int rank, const char* file, int line);
};

template <typename... Args>
void Logger::log(Level level, const char* file, int line, const char* format, const Args&... args)
{
    if (!IsEnabled(level))
    {
        return;
    }
    std::lock_guard<std::mutex> lock(logMutex);
#ifdef TM_LOG_PRINT_RANK
    std::string message = GetPrefix(level, rank_, file, line) + fmtstr(format, args...);
#else
    std::string message = GetPrefix(level, file, line) + fmtstr(format, args...);
#endif

    LogStream(level < Level::WARNING ? std::cout : std::cerr) << message;
    if (outFile.is_open())
    {
        outFile << message << std::endl;
    }
}

template <typename... Args>
void Logger::log(Level level, const char* file, int line, const std::string& format, const Args&... args)
{
    log(level, file, line, format.c_str(), args...);
}

#ifdef TM_LOG_ENABLE
#define LOG_TRACE(format, ...) Logger::GetInstance().log(Logger::TRACE, __FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_DEBUG(format, ...) Logger::GetInstance().log(Logger::DEBUG, __FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_INFO(format, ...) Logger::GetInstance().log(Logger::INFO, __FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_WARNING(format, ...) Logger::GetInstance().log(Logger::WARNING, __FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_ERROR(format, ...) Logger::GetInstance().log(Logger::ERROR, __FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_EXCEPTION(ex, ...) Logger::GetInstance().log(ex, Logger::ERROR)
#else
#define LOG_TRACE(format, ...)
#define LOG_DEBUG(format, ...)
#define LOG_INFO(format, ...)
#define LOG_WARNING(format, ...)
#define LOG_ERROR(format, ...)
#endif
