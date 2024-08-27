
#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <complex>

#include <fstream>
#include <string>
#include <sstream>

#include <vector>

class TriangleMesh {
public:
	TriangleMesh();
	void draw();
	~TriangleMesh();

private:
	unsigned int EBO, VAO, element_count;
	std::vector<unsigned int> VBOs;
};