# CraftBeerPi4 RIMS PID AutoTune Plugin

This plugin implements a specialized PID autotune system for RIMS (Recirculating Infusion Mash System), specifically focusing on MLT (Mash/Lauter Tun) temperature control.

## Features

- Adaptive autotune for RIMS systems
- RIMS and MLT temperature monitoring
- Recirculation flow control
- Automatic power adjustment based on temperature difference
- Optimized PID parameters for RIMS systems

## Installation

1. Install the plugin via CBPI4 Web Interface
2. Restart CBPI4

## Parameters

- **Output_Step**: Output power when stepping up/down (Default: 100%)
- **Max_Output**: Maximum output power (Default: 100%)
- **lockback_seconds**: Time to analyze min/max temps (Default: 30s)
- **rims_flow_threshold**: Minimum flow for RIMS (Default: 1.0 L/min)
- **max_temp_difference**: Maximum RIMS/MLT difference (Default: 5°C)
- **safety_margin**: Safety margin (Default: 2°C)
- **RIMS_Sensor**: RIMS temperature sensor
- **Flow_Sensor**: Optional flow sensor
- **SampleTime**: Sample time (Default: 5s)

## Usage

1. Configure the plugin as Kettle Logic
2. Set up RIMS and MLT sensors
3. Configure flow sensor (optional)
4. Set target temperature
5. Start autotune and wait for completion

## Tuning Rules

- **rims-aggressive**: For systems needing fast response
- **rims-moderate**: Balance between response and stability
- **rims-conservative**: For systems needing more stability

## Safety Features

- Continuous RIMS/MLT temperature difference monitoring
- Automatic power adjustment to prevent overheating
- Minimum flow verification
- Automatic shutdown in unsafe conditions

## Changelog

- 1.0.0: Initial Release
  - Implementation of adaptive autotune for RIMS
  - Addition of RIMS-specific tuning rules
  - RIMS temperature-based safety controls

## License

GNU General Public License v3 