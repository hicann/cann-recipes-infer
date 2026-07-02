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
#include "logger.h"
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>

Logger::Logger()
{

    auto getEnvVar = [](const char* name, const char* defaultValue) -> std::string
    {
        if (const char* value = std::getenv(name))
        {
            return value;
        }
        return defaultValue;
    };
    std::string levelName = getEnvVar("TM_LOG_LEVEL", "INFO");

    auto level = [levelName]()
    {
        static const std::unordered_map<std::string, Level> levelMap = {{"TRACE", Level::TRACE},
            {"DEBUG", Level::DEBUG}, {"INFO", Level::INFO}, {"WARNING", Level::WARNING}, {"ERROR", Level::ERROR}};
        auto it = levelMap.find(levelName);
        return it != levelMap.end() ? it->second : Level::INFO;
    }();

    SetLevel(level);
}

Logger::~Logger()
{
    if (outFile.is_open())
    {
        outFile.close();
    }
}

Logger& Logger::GetInstance()
{
    static Logger instance;
    return instance;
}

void Logger::SetLogFilePath(const std::string& dir)
{
    {
        std::lock_guard<std::mutex> lock(logMutex);
        outputDir = dir;
        if (!outputDir.empty() && outputDir.back() != '/')
        {
            outputDir += '/';
        }

        if (outFile.is_open())
        {
            outFile.close();
        }

        std::string logfile
            = outputDir + "log_" + std::to_string(std::time(nullptr)) + "_rank_" + std::to_string(rank_) + ".log";
        outFile.open(logfile, std::ios::app);
        if (!outFile.is_open())
        {
            std::cerr << "Failed to open log file: " << logfile << std::endl;
        }
    }

    log(INFO, __FILE__, __LINE__, "Set output directory to %s", outputDir.c_str());
}

const std::string& Logger::GetOutputDir() const
{
    return outputDir;
}

void Logger::SetRank(int rank)
{
    rank_ = rank;
}

int Logger::GetRank() const
{
    return rank_;
}

void Logger::log(std::exception& ex, Level level)
{
    std::lock_guard<std::mutex> lock(logMutex);
    std::string message = GetPrefix(level) + "Exception: " + ex.what();
    LogStream(level < Level::WARNING ? std::cout : std::cerr) << message;

    if (outFile.is_open())
    {
        outFile << message << std::endl;
    }
}

Logger::Level Logger::GetLevel() const
{
    return level_;
}

void Logger::SetLevel(const Level level)
{
    level_ = level;
    log(INFO, __FILE__, __LINE__, "Set logger level to %s", GetLevelName(level));
}

bool Logger::IsEnabled(const Level level) const
{
    return level_ <= level;
}

const char* Logger::GetLevelName(const Level level)
{
    switch (level)
    {
        case TRACE: return "TRACE";
        case DEBUG: return "DEBUG";
        case INFO: return "INFO";
        case WARNING: return "WARNING";
        case ERROR: return "ERROR";
    }
    throw std::runtime_error(fmtstr("Unknown log level: %d", static_cast<int>(level)));
}

std::string Logger::GetPrefix(const Level level)
{
    return fmtstr("[%s] ", GetLevelName(level));
}

std::string Logger::GetPrefix(const Level level, const int rank)
{
    return fmtstr("[%s] [%d] ", GetLevelName(level), rank);
}

std::string Logger::GetPrefix(const Level level, const char* file, int line)
{
    const char* filename = strrchr(file, '/');
    filename = filename ? filename + 1 : file;
    return fmtstr("[%s] [%s:%d] ", GetLevelName(level), filename, line);
}

std::string Logger::GetPrefix(const Level level, int rank, const char* file, int line)
{
    const char* filename = strrchr(file, '/');
    filename = filename ? filename + 1 : file;
    return fmtstr("[%s] [%d] [%s:%d] ", GetLevelName(level), rank, filename, line);
}
