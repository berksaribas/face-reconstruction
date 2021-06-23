#pragma once

#include "points.h"

template<typename PointType>
std::vector<PointType> read_points_from_file(const std::string& file_path)
{
    std::ifstream file(file_path);

    if(!file.is_open())
    {
		throw std::runtime_error("Failed to open data file: " + file_path);
    }

    std::vector<PointType> points{};

    std::string line{};
    while(std::getline(file, line))
    {
        const auto point = PointType::from_string(line);
		points.push_back(point);
    }

	return points;
}
