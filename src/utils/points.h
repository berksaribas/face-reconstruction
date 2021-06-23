#pragma once

#include <vector>
#include <fstream>
#include <istream>
#include <iostream>
#include <sstream>

struct Weight
{
	Weight() = default;

	Weight(double w_) : 
		w{ w_ }
	{
	}

	static Weight from_string(const std::string& line)
	{
		std::istringstream stream(line);
		double w{};
		stream >> w;

		Weight weight{ w };
		return weight;
	}

	double w{};
};


struct Point2D
{
	Point2D() = default;
	
	Point2D(double x_, double y_)
		: x{ x_ }, y{ y_ }
	{
	}

	static Point2D from_string(const std::string& line)
	{
		std::istringstream stream(line);
		float x{};
		float y{};
		stream >> x >> y;

		Point2D point{ x, y };
		return point;
	}

	double x{};
	double y{};
};

struct Point3D
{
	Point3D(double x_, double y_, double z_)
		: x{ x_ }, y{ y_ }, z{ z_ }
	{
	}

	static Point3D from_string(const std::string& line)
	{
		std::istringstream stream(line);
		float x{};
		float y{};
		float z{};
		stream >> x >> y >> z;

		Point3D point{ x, y, z };
		return point;
	}

	double x{};
	double y{};
	double z{};
};

using PointList2D = std::vector<Point2D>;
using PointList3D = std::vector<Point3D>;