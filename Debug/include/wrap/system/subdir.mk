################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../include/wrap/system/getopt.cpp 

OBJS += \
./include/wrap/system/getopt.o 

CPP_DEPS += \
./include/wrap/system/getopt.d 


# Each subdirectory must supply rules for building sources it contributes
include/wrap/system/%.o: ../include/wrap/system/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/home/khaque/opencv2.4/include -I/home/khaque/workspace/4DReconstruction/include -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


