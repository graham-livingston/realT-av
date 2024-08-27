// CMakeProject1.cpp : Defines the entry point for the application.
//

#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <complex>
#include "RtAudio.h"
#include "rtaudio_c.h"
#include "pocketfft_hdronly.h"
#include <mutex>
#include "triangle_mesh.h"
#include "RtMidi.h"

#include <fstream>
#include <string>
#include <sstream>

#include <vector>
#include <cmath>
#include <exception>
#include <unordered_map>
#include <utility> // For std::pair
#include <algorithm> // For std::sort

#include <unordered_map>

using namespace std;
using namespace pocketfft;

void keyboardInputParser(GLFWwindow* window, int key, int scancode, int action, int mods); // Function prototype for keyboard input parser
void toggleFullscreen(GLFWwindow* window); // Function prototype for toggling fullscreen mode

unsigned int make_shader(const std::string& vertex_filepath, const std::string& fragment_filepath);
unsigned int make_module(const std::string& filepath, unsigned int module_type);
std::vector<std::complex<float>> performFFT(const std::vector<float>& audioData);
std::vector<float> performFFTAndGetMagnitudes(const std::vector<float>& audioData);
float calculateAverageMagnitude(const std::vector<float>& fftMagnitudes);
int audioCallback(void* outputBuffer, void* inputBuffer, unsigned int nBufferFrames,
	double streamTime, RtAudioStreamStatus status, void* userData);
void printFFTMagnitudes(const std::vector<float>& magnitudes);


// Declaration for countNotesInRanges
float countNotesInRanges(const std::vector<float>& fftMagnitudes, const std::vector<std::pair<float, float>>& frequencyRanges);
// Declaration for countMagnitudesInRanges
float countMagnitudesInRanges(const std::vector<float>& fftMagnitudes, const std::vector<std::pair<float, float>>& frequencyRanges);
void calculateMeanAndMedianMagnitudes(const std::vector<float>& fftMagnitudes, const std::vector<std::pair<float, float>>& frequencyRanges, std::vector<float>& means, std::vector<float>& medians);

// Declaration for integrateDataForUniform
void integrateDataForUniform(const std::vector<float>& fftMagnitudes, const std::vector<std::pair<float, float>>& frequencyRanges, std::vector<std::tuple<float, float, float, float, float>>& integratedData);

// Declaration for Normalized Ranking
void calculateMeanMagnitudeRanking(const std::vector<std::pair<float, float>>& frequencyRanges, const std::vector<float>& means, std::vector<float>& rankings);



// Global variables
std::mutex fftDataMutex; // Existing mutex for thread-safe access
const size_t numMagnitudes = 128; // Example: number of magnitudes you want to pass
std::vector<float> fftMagnitudesGlobal(numMagnitudes, 0.0f); // Global variable to store magnitudes
std::vector<int> fftMagnitudesLocations(numMagnitudes);
unsigned int shader; // Global variable for the shader program ID
GLint numpadValueLocation; // Global variable for the uniform location
float numpadValue = -1.0; // Global variable for the numpad value


// Midi variables
RtMidiIn* midiin = 0;
std::vector<unsigned char> message;
int nBytes, i;
double stamp;


// Global variables for color handling
std::vector<std::string> activeColors;
std::vector<bool> activeNotes(128, false); // Global variable to track active notes
int currentColorIndex = -1;

// Define a map from MIDI note numbers to color indices
std::unordered_map<int, int> noteToColorMap = {
	{0,0}, 
	{52, 1}, {53, 2}, {54, 3}, {55, 4}, //white
	{56, 5}, {57, 3}, {58, 4}, // and white
	{59, 5},{60, 3}, {61, 4}, {62, 5}, //pink
	{63, 6}, {64,6}, {65,6}, //and pink
	{66, 7}, {67, 7}, {68, 7}, {69, 7}, //red
	{70, 8}, {71, 8}, {72, 8}, {73, 8}, //and red
	{74, 9}, {75, 9}, {76, 9}, {77, 9}, //grey
	{78, 10}, {79, 10}, {80, 10}, {81, 10}, //blue
	{82, 11}, {83, 11}, {84, 11}, //and blue
	{85, 12}, {86, 12}, {87, 12}, {88, 12}, //putple
	{89, 13}, {90, 13}, {91, 13}, //and purple 
	{92, 14}, {93, 14}, {94, 14}, {95, 14}, //violet
	{96, 15}, {97, 15}, {98, 15}, //and violet
	{100, 16}, {101, 16}, {102, 16}, {103, 16}, //brown
	{104, 17}, {105, 17}, {106, 17}, //and brown
	{108, 18}, {109, 18}, {110, 18}, {111, 18}, //black
	{112, 19}, {113, 19}, {114, 19}, //and black
	{115, 20}, {116, 20}, {117, 20}, //quantum
	{118, 21}, {119, 21}, {120, 21}, //quantum
	{121, 22}, {122, 22}, {123, 22}, //and quantum
	{124, 23}, {125, 23}, {126, 23},  //and grey
	{127,24} //grey question
};

std::vector<float> colorToRGB(const std::string& color) {
	if (color == "white") return { 1.0f, 1.0f, 1.0f };
	// Add other color mappings
	return { 0.0f, 0.0f, 0.0f }; // Default color
}

// MIDI callback function declaration
void mycallback(double deltatime, std::vector<unsigned char>* message, void* userData);




int main()
{
	// Initialize window
	std::ifstream file;
	std::string line;

	if (!glfwInit())
	{
		return -1;
	}
	GLFWwindow* window = glfwCreateWindow(2560, 1440, "Hello World", NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);


	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		cout << "Failed to initialize GLAD" << endl;
		return -1;
	}

	// Initialize RtAudio
	RtAudio adc;
	if (adc.getDeviceCount() < 1) {
		std::cerr << "\nNo audio devices found!\n";
		exit(1);
	}

	RtAudio::StreamParameters parameters;
	parameters.deviceId = adc.getDefaultInputDevice();
	parameters.nChannels = 1; // Mono input
	parameters.firstChannel = 0;
	unsigned int sampleRate = 44100;
	unsigned int bufferFrames = 512; // Adjust as needed

	RtAudioErrorType result = adc.openStream(nullptr, &parameters, RTAUDIO_FLOAT32, sampleRate, &bufferFrames, &audioCallback);
	if (result != RTAUDIO_NO_ERROR) {
		std::cerr << "Failed to open stream: " << adc.getErrorText() << std::endl;
		exit(1);
	}

	result = adc.startStream();
	if (result != RTAUDIO_NO_ERROR) {
		std::cerr << "Failed to start stream: " << adc.getErrorText() << std::endl;
		exit(1);
	}

	// Initialize MIDI
	// RtMidiIn constructor
	try {
		midiin = new RtMidiIn();
	}
	catch (RtMidiError& error) {
		error.printMessage();
		exit(EXIT_FAILURE);
	}

	// Check available ports.
	unsigned int nPorts = midiin->getPortCount();
	if (nPorts == 0) {
		std::cout << "No MIDI ports available!\n";
		exit(0);
	}

	midiin->openPort(0);
	midiin->setCallback(&mycallback);
	midiin->ignoreTypes(false, false, false);

	// Initialize Frequency Ranges
	std::vector<std::pair<float, float>> frequencyRanges = {
		{0.0, 1049.9999333651401}, // Bin 1
		{1049.9999333651401, 2099.9998667302802}, // Bin 2
		{2099.9998667302802, 3149.99980009542}, // Bin 3
		{3149.99980009542, 4199.9997334605603}, // Bin 4
		{4199.9997334605603, 5249.9996668257}, // Bin 5
		{5249.9996668257, 6299.99960019084}, // Bin 6
		{6299.99960019084, 7349.99953355598}, // Bin 7
		{7349.99953355598, 8399.99946692112}, // Bin 8
		{8399.99946692112, 9449.99940028626}, // Bin 9
		{9449.99940028626, 10499.9993336514}, // Bin 10
		{10499.9993336514, 11549.9992670165}, // Bin 11
		{11549.9992670165, 12599.9992003817}, // Bin 12
		{12599.9992003817, 13649.9991337468}, // Bin 13
		{13649.9991337468, 14699.999067112}, // Bin 14
		{14699.999067112, 15749.9990004771}, // Bin 15
		{15749.9990004771, 16799.9989338423}, // Bin 16
		{16799.9989338423, 17849.9988672074}, // Bin 17
		{17849.9988672074, 18899.9988005726}, // Bin 18
		{18899.9988005726, 19949.9987339377}, // Bin 19
		{19949.9987339377, 20999.9986673029}, // Bin 20
		{20999.9986673029, 22049.998600668} // Bin 21														
	};

	// Integrate data for uniform passing
	std::vector<std::tuple<float, float, float, float, float>> integratedData;
	integrateDataForUniform(fftMagnitudesGlobal, frequencyRanges, integratedData);


	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	TriangleMesh* triangle = new TriangleMesh();

	unsigned int shader = make_shader("shaders/vertex.txt", "shaders/fragment.txt");
	//numpadValueLocation = glGetUniformLocation(shader, "u_numpadValue");

	// Register keyboard input callback
	glfwSetKeyCallback(window, keyboardInputParser);

	for (size_t i = 0; i < numMagnitudes; ++i) {
		std::string uniformName = "fftMagnitudes[" + std::to_string(i) + "]";
		fftMagnitudesLocations[i] = glGetUniformLocation(shader, uniformName.c_str());
	}

	// Get the location of the uniform variable in the shader
	GLint fftAverageLocation = glGetUniformLocation(shader, "u_fftAverage");






	while (!glfwWindowShouldClose(window))
	{

		glfwPollEvents();

		// Pass the window size to the shader
		int resolutionLocation = glGetUniformLocation(shader, "u_resolution");
		// Get the current window size
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);

		glUseProgram(shader);
		glUniform2f(resolutionLocation, (float)width, (float)height);
		float timeValue = glfwGetTime(); // Get elapsed time
		int timeLocation = glGetUniformLocation(shader, "u_time");
		glUniform1f(timeLocation, timeValue); // Pass time to the shader
		GLint numpadValueLocation = glGetUniformLocation(shader, "u_numpadValue");
		glUniform1f(numpadValueLocation, numpadValue);

		if (currentColorIndex != -1) {
			GLint colorIndexLocation = glGetUniformLocation(shader, "u_colorIndex");
			glUniform1i(colorIndexLocation, currentColorIndex);
		}


		// Lock the mutex and read the shared fftMagnitudesGlobal
		float averageMagnitude;
		{
			std::lock_guard<std::mutex> lock(fftDataMutex);
			averageMagnitude = calculateAverageMagnitude(fftMagnitudesGlobal);
			for (size_t i = 0; i < numMagnitudes; ++i) {
				glUniform1f(fftMagnitudesLocations[i], fftMagnitudesGlobal[i]);
			}
		}

		// Update the uniform variable with the average magnitude
		glUniform1f(fftAverageLocation, averageMagnitude);

		// Pass the integrated data to the fragment shader
		for (size_t i = 0; i < integratedData.size(); ++i) {
			// Construct the uniform names
			std::string uniformNameNotesCount = "integratedData[" + std::to_string(i) + "].notesCount";
			std::string uniformNameMagnitudesCount = "integratedData[" + std::to_string(i) + "].magnitudesCount";
			std::string uniformNameMean = "integratedData[" + std::to_string(i) + "].mean";
			std::string uniformNameMedian = "integratedData[" + std::to_string(i) + "].median";

			// Get the location of the uniform variables
			GLint locationNotesCount = glGetUniformLocation(shader, uniformNameNotesCount.c_str());
			GLint locationMagnitudesCount = glGetUniformLocation(shader, uniformNameMagnitudesCount.c_str());
			GLint locationMean = glGetUniformLocation(shader, uniformNameMean.c_str());
			GLint locationMedian = glGetUniformLocation(shader, uniformNameMedian.c_str());

			// Set the uniform variables
			glUniform1i(locationNotesCount, std::get<0>(integratedData[i]));
			glUniform1i(locationMagnitudesCount, std::get<1>(integratedData[i]));
			glUniform1f(locationMean, std::get<2>(integratedData[i]));
			glUniform1f(locationMedian, std::get<3>(integratedData[i]));
		}

		glClear(GL_COLOR_BUFFER_BIT);

		triangle->draw();

		glfwSwapBuffers(window);
	}

	// Cleanup
	if (adc.isStreamOpen()) {
		adc.stopStream(); // stopStream() and closeStream() do not throw exceptions
	}
	adc.closeStream();
	delete midiin;
	glDeleteProgram(shader);
	glfwTerminate();
	return 0;
}


// function for making the shader program

unsigned int make_shader(const std::string& vertex_filepath, const std::string& fragment_filepath) {

	std::vector<unsigned int> modules;
	modules.push_back(make_module(vertex_filepath, GL_VERTEX_SHADER));
	modules.push_back(make_module(fragment_filepath, GL_FRAGMENT_SHADER));

	unsigned int shader = glCreateProgram();
	for (unsigned int shaderModule : modules) {
		glAttachShader(shader, shaderModule);
	}
	glLinkProgram(shader);

	int success;
	glGetShaderiv(shader, GL_LINK_STATUS, &success);
	if (!success) {
		char errorLog[1024];
		glGetShaderInfoLog(shader, 1024, NULL, errorLog);
		std::cout << "ERROR::SHADER::Linking error:\n" << errorLog << std::endl;
	}

	for (unsigned int shaderModule : modules) {
		glDeleteShader(shaderModule);
	}

	return shader;
}

// function for making a shader module
unsigned int make_module(const std::string& filepath, unsigned int module_type) {

	std::ifstream file;
	std::stringstream bufferedLines;
	std::string line;

	file.open(filepath);
	while (std::getline(file, line))
	{
		bufferedLines << line << "\n";
	}
	std::string shaderSource = bufferedLines.str();
	const char* shaderSrc = shaderSource.c_str();
	bufferedLines.str("");
	file.close();

	unsigned int shaderModule = glCreateShader(module_type);
	glShaderSource(shaderModule, 1, &shaderSrc, NULL);
	glCompileShader(shaderModule);

	int success;
	glGetShaderiv(shaderModule, GL_COMPILE_STATUS, &success);
	if (!success) {
		char errorLog[1024];
		glGetShaderInfoLog(shaderModule, 1024, NULL, errorLog);
		std::cout << "ERROR::SHADER::COMPILATION_FAILED\n" << errorLog << std::endl;
	}

	return shaderModule;
}


// Function to perform FFT on audio data
std::vector<std::complex<float>> performFFT(const std::vector<float>& audioData) {
	size_t N = audioData.size();

	// PocketFFT expects input for r2c transforms as std::vector<float>
	// and provides output as std::vector<std::complex<float>>
	std::vector<std::complex<float>> fftOutput(N / 2 + 1);

	// Define the shape of the data
	shape_t shape = { N };

	// Define strides for input and output
	stride_t stride_in = { sizeof(float) };
	stride_t stride_out = { sizeof(complex<float>) };

	// Define the axes along which to perform the FFT
	shape_t axes = { 0 };

	// Perform the real-to-complex FFT
	r2c(shape, stride_in, stride_out, axes, true, audioData.data(), fftOutput.data(), 1.0f);

	return fftOutput;
}


// Function to perform FFT on audio data and return the magnitudes
std::vector<float> performFFTAndGetMagnitudes(const std::vector<float>& audioData) {
	using namespace pocketfft;
	size_t N = audioData.size();

	// Perform the FFT
	std::vector<std::complex<float>> fftOutput = performFFT(audioData);

	// Prepare a vector to hold the magnitudes
	std::vector<float> magnitudes(fftOutput.size());

	// Compute the magnitudes
	for (size_t i = 0; i < fftOutput.size(); ++i) {
		magnitudes[i] = std::abs(fftOutput[i]);
	}

	return magnitudes;
}

// Function to calculate the average magnitude from the FFT data
float calculateAverageMagnitude(const std::vector<float>& fftMagnitudes) {
	float averageMagnitude = 0.0f;
	for (float mag : fftMagnitudes) {
		averageMagnitude += mag;
	}
	averageMagnitude /= fftMagnitudes.size();
	return averageMagnitude;
}

// Callback function to handle incoming audio data
int audioCallback(void* outputBuffer, void* inputBuffer, unsigned int nBufferFrames,
	double streamTime, RtAudioStreamStatus status, void* userData) {
	if (status) std::cerr << "Stream overflow detected!" << std::endl;

	// Cast inputBuffer to the type used by your audio data (e.g., float)
	auto* inBuffer = static_cast<float*>(inputBuffer);

	// Perform FFT on the audio data
	std::vector<float> audioChunk(inBuffer, inBuffer + nBufferFrames);

	// Calculate magnitudes from FFT
	std::vector<float> fftMagnitudes = performFFTAndGetMagnitudes(audioChunk);

	// Sample or process fftMagnitudes as needed, then store in global variable
	{
		std::lock_guard<std::mutex> lock(fftDataMutex);
		for (size_t i = 0; i < numMagnitudes; ++i) {
			// Simple resampling example
			size_t index = i * fftMagnitudes.size() / numMagnitudes;
			fftMagnitudesGlobal[i] = fftMagnitudes[index];
		}
	}
	// Calculate the average magnitude from the FFT data
	//float localAverageMagnitude = calculateAverageMagnitude(fftMagnitudes);


	//Lock the mutex and update the shared fftAverage variable
	//std::lock_guard<std::mutex> lock(fftDataMutex);
	//fftAverage = localAverageMagnitude;

	return 0; // Return 0 to continue streaming
}
// Function to print the FFT magnitudes to the console
void printFFTMagnitudes(const std::vector<float>& magnitudes) {
	std::stringstream ss;
	for (size_t i = 0; i < magnitudes.size(); ++i) {
		ss << magnitudes[i];
		if (i < magnitudes.size() - 1) {
			ss << ", ";
		}
	}
	std::cout << ss.str() << std::endl;
}

float countNotesInRanges(const std::vector<float>& fftMagnitudes, const std::vector<std::pair<float, float>>& frequencyRanges) {
	float totalNotes = 0.0f;

	// Iterate through each frequency range
	for (const auto& range : frequencyRanges) {
		float notesCount = 0.0f;

		// Iterate through the FFT magnitudes to find peaks within the current range
		for (size_t i = 1; i < fftMagnitudes.size() - 1; ++i) {
			// Check if the current magnitude is a peak and falls within the current frequency range
			if (fftMagnitudes[i] > fftMagnitudes[i - 1] && fftMagnitudes[i] > fftMagnitudes[i + 1] &&
				fftMagnitudes[i] >= range.first && fftMagnitudes[i] < range.second) {
				notesCount += 1.0f;
			}
		}

		totalNotes += notesCount;
	}

	return totalNotes;
}


float countMagnitudesInRanges(const std::vector<float>& fftMagnitudes, const std::vector<std::pair<float, float>>& frequencyRanges) {
	float totalMagnitudes = 0.0f;

	// Iterate through each frequency range
	for (const auto& range : frequencyRanges) {
		float magnitudesCount = 0.0f;

		// Iterate through the FFT magnitudes to count how many fall within the current range
		for (const auto& magnitude : fftMagnitudes) {
			if (magnitude >= range.first && magnitude < range.second) {
				magnitudesCount += 1.0f;
			}
		}

		totalMagnitudes += magnitudesCount;
	}

	return totalMagnitudes;
}

// Function to calculate the mean and median magnitudes for each frequency range
void calculateMeanAndMedianMagnitudes(const std::vector<float>& fftMagnitudes, const std::vector<std::pair<float, float>>& frequencyRanges, std::vector<float>& means, std::vector<float>& medians) {
	// Initialize means and medians vectors with default values of 0.0
	means.assign(frequencyRanges.size(), 0.0f);
	medians.assign(frequencyRanges.size(), 0.0f);

	for (size_t rangeIndex = 0; rangeIndex < frequencyRanges.size(); ++rangeIndex) {
		std::vector<float> magnitudesInCurrentRange;

		// Collect all magnitudes within the current frequency range
		for (const auto& magnitude : fftMagnitudes) {
			if (magnitude >= frequencyRanges[rangeIndex].first && magnitude < frequencyRanges[rangeIndex].second) {
				magnitudesInCurrentRange.push_back(magnitude);
			}
		}

		// Only calculate mean and median if there are magnitudes in the current range
		if (!magnitudesInCurrentRange.empty()) {
			// Calculate mean
			float sum = 0.0f;
			for (const auto& mag : magnitudesInCurrentRange) {
				sum += mag;
			}
			means[rangeIndex] = sum / magnitudesInCurrentRange.size();

			// Calculate median
			std::sort(magnitudesInCurrentRange.begin(), magnitudesInCurrentRange.end());
			size_t midIndex = magnitudesInCurrentRange.size() / 2;
			medians[rangeIndex] = magnitudesInCurrentRange.size() % 2 == 0 ?
				(magnitudesInCurrentRange[midIndex] + magnitudesInCurrentRange[midIndex - 1]) / 2.0f :
				magnitudesInCurrentRange[midIndex];
		}
	}
}


void integrateDataForUniform(const std::vector<float>& fftMagnitudes, const std::vector<std::pair<float, float>>& frequencyRanges, std::vector<std::tuple<float, float, float, float, float>>& integratedData) {
	integratedData.clear();
	std::vector<float> means, medians, rankings;

	// Calculate the counts, means, and medians for each frequency range
	calculateMeanAndMedianMagnitudes(fftMagnitudes, frequencyRanges, means, medians);

	// Calculate the rankings based on mean magnitudes
	calculateMeanMagnitudeRanking(frequencyRanges, means, rankings);

	// Integrate all the data into a single structure
	for (size_t i = 0; i < frequencyRanges.size(); ++i) {
		int notesCount = countNotesInRanges(fftMagnitudes, { frequencyRanges[i] });
		int magnitudesCount = countMagnitudesInRanges(fftMagnitudes, { frequencyRanges[i] });
		integratedData.push_back(std::make_tuple(notesCount, magnitudesCount, means[i], medians[i], rankings[i]));
	}
}


void calculateMeanMagnitudeRanking(const std::vector<std::pair<float, float>>& frequencyRanges, const std::vector<float>& means, std::vector<float>& rankings) {
	// Ensure the rankings vector has the correct size
	rankings.resize(frequencyRanges.size(), 0.0f);

	// Find the minimum and maximum mean magnitudes
	auto minMaxIt = std::minmax_element(means.begin(), means.end());
	float minMean = *minMaxIt.first;
	float maxMean = *minMaxIt.second;
	float range = maxMean - minMean;

	// Handle the case where all means are equal (avoid division by zero)
	if (range == 0.0f) {
		std::fill(rankings.begin(), rankings.end(), 1.0f); // Assign equal ranking
		return;
	}

	// Calculate rankings based on mean magnitudes
	for (size_t i = 0; i < means.size(); ++i) {
		rankings[i] = (means[i] - minMean) / range; // Normalize to [0.0, 1.0]
	}
}

inline void printNumpadValue(float value) {
	std::cout << "Value: " << value << std::endl;
}

void keyboardInputParser(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action == GLFW_PRESS) {
		switch (key) {
		case GLFW_KEY_KP_0: numpadValue = 0.0; break;
		case GLFW_KEY_KP_1: numpadValue = 1.0; break;
		case GLFW_KEY_KP_2: numpadValue = 2.0; break;
		case GLFW_KEY_F: toggleFullscreen(window); break;
			// Add cases for other numpad keys as needed
		default: break; // Default case for no key pressed
		}
	}
	
	// Update the uniform variable in the shader
	glUseProgram(shader);
	glUniform1f(numpadValueLocation, numpadValue);
}

void toggleFullscreen(GLFWwindow* window) {
	static bool isFullscreen = false;
	static int windowedWidth, windowedHeight, windowedX, windowedY;
	static GLFWmonitor* lastMonitor = nullptr;

	if (!isFullscreen) {
		// Store the windowed mode position and size
		glfwGetWindowPos(window, &windowedX, &windowedY);
		glfwGetWindowSize(window, &windowedWidth, &windowedHeight);

		// Get the current monitor based on the window's position
		int monitorCount;
		GLFWmonitor** monitors = glfwGetMonitors(&monitorCount);
		GLFWmonitor* currentMonitor = nullptr;
		int mx, my, mWidth, mHeight;
		for (int i = 0; i < monitorCount; ++i) {
			glfwGetMonitorWorkarea(monitors[i], &mx, &my, &mWidth, &mHeight);
			if (windowedX >= mx && windowedX < (mx + mWidth) &&
				windowedY >= my && windowedY < (my + mHeight)) {
				currentMonitor = monitors[i];
				break;
			}
		}

		if (currentMonitor) {
			lastMonitor = currentMonitor;
			const GLFWvidmode* mode = glfwGetVideoMode(currentMonitor);
			glfwSetWindowMonitor(window, currentMonitor, 0, 0, mode->width, mode->height, mode->refreshRate);
			glfwSetWindowAttrib(window, GLFW_DECORATED, GLFW_FALSE);
		}
	}
	else {
		// Restore the window to windowed mode with decorations
		glfwSetWindowMonitor(window, nullptr, windowedX, windowedY, windowedWidth, windowedHeight, GLFW_DONT_CARE);
		glfwSetWindowAttrib(window, GLFW_DECORATED, GLFW_TRUE);
	}

	isFullscreen = !isFullscreen;
}


// Function to handle incoming MIDI messages
	void mycallback(double deltatime, std::vector<unsigned char>*message, void* userData) {
		unsigned int nBytes = message->size();
		int note = message->at(1);
		int velocity = message->at(2);

		// Print MIDI message details
		std::cout << "MIDI Message Received:" << std::endl;
		std::cout << " Timestamp: " << deltatime << "s" << std::endl;
		std::cout << " Message Size: " << nBytes << " bytes" << std::endl;
		for (unsigned int i = 0; i < nBytes; i++) {
			std::cout << " Byte " << i << ": " << static_cast<int>(message->at(i)) << std::endl;
		}

		// Note on event
		if ((message->at(0) & 0xF0) == 0x90 && velocity > 0) {
			std::cout << " Note On: " << note << " Velocity: " << velocity << std::endl;
			// Additional logic for handling Note On event
			auto search = noteToColorMap.find(note);
			if (search != noteToColorMap.end()) {
				currentColorIndex = search->second;
			}
		}
		// Note off event
		else if (((message->at(0) & 0xF0) == 0x80) || ((message->at(0) & 0xF0) == 0x90 && velocity == 0)) {
			std::cout << " Note Off: " << note << " Velocity: " << velocity << std::endl;
			// Additional logic for handling Note Off event
			auto search = noteToColorMap.find(note);
			if (search != noteToColorMap.end()) {
				currentColorIndex = 0;
			}
		}
	}


