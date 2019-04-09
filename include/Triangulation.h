#pragma once
#include "Cube.h"
#include "DataStructs.h"

namespace MeshReconstruction
{
	void Triangulate(
		IntersectInfo const& intersect,
		//Fun3v const& grad,
		Mesh& mesh, Vec3 const& position, cv::Mat voxel3D);
}
