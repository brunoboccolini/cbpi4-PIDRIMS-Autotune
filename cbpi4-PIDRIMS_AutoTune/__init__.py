# PIDRIMS AutoTune Plugin for CraftBeerPi 4
# Based on the original PID AutoTune with modifications for RIMS systems
# This plugin implements PID autotuning specifically for RIMS (Recirculating Infusion Mash System) brewing systems
# It includes safety features to prevent the RIMS from overheating during the autotuning process
# Author: Bruno Boccolini
# License: GNU General Public License v3

from time import localtime, strftime, time
import math
import logging
from collections import deque, namedtuple
import asyncio
from cbpi.api import *
from cbpi.api.dataclasses import NotificationAction, NotificationType

# Plugin configuration parameters
# These parameters can be adjusted through the CraftBeerPi interface
@parameters([Property.Number(label = "Output_Step", configurable = True, default_value = 100, description="Default: 100. Sets the output power when stepping up/down."),
             Property.Number(label = "Max_Output", configurable = True, default_value = 100, description="Default: 100. Sets the maximum output power."),
             Property.Number(label = "lookback_seconds", configurable = True, default_value = 30, description="Default: 30. Time in seconds to look for min/max temperatures."),
             Property.Number(label = "max_temp_difference", configurable = True, default_value = 5, description="Default: 5. Maximum temperature difference between RIMS and target temp (°C)."),
             Property.Sensor(label="RIMS_Sensor", description="Sensor for measuring RIMS temperature"),
             Property.Actor(label="Pump", description="Pump actor for RIMS recirculation"),
             Property.Number(label = "SampleTime", configurable = True, default_value = 5, description="Default: 5. Sample time in seconds for PID calculation.")])

class PIDRIMSAutotune(CBPiKettleLogic):
    # Main class for RIMS PID Autotuning
    # Implements safety checks and dynamic parameter adjustment based on system behavior

    def __init__(self, cbpi, id, props):
        # Initialize the AutoTune plugin with necessary parameters and state variables
        super().__init__(cbpi, id, props)
        self._logger = logging.getLogger(type(self).__name__)
        # Get configuration from properties
        self.rims_sensor = self.props.get("RIMS_Sensor", None)
        self.pump = self.props.get("Pump", None)
        self.sample_time = float(self.props.get("SampleTime", 5))
        self.max_temp_diff = float(self.props.get("max_temp_difference", 5))
        # Initialize state flags
        self.finished = False
        self.running = False
        self.auto_mode = False
        self.heater = None
        self.kettle = None

    async def on_start(self):
        # Chamado quando o Auto Mode é ligado
        try:
            self.kettle = self.get_kettle(self.id)
            self.heater = self.kettle.heater
            
            # Verifica se a bomba está configurada
            if not self.pump:
                self.cbpi.notify('PIDRIMS AutoTune', 'Bomba não configurada!', NotificationType.ERROR)
                await self.stop()
                return

            # Liga a bomba primeiro
            await self.actor_on(self.pump, 100)
            await asyncio.sleep(2)  # Aguarda a bomba estabilizar

            # Verifica se a bomba está funcionando
            if self.cbpi.actor.find_by_id(self.pump).power <= 0:
                self.cbpi.notify('PIDRIMS AutoTune', 'Falha ao ligar a bomba!', NotificationType.ERROR)
                await self.stop()
                return

            # Se chegou aqui, a bomba está funcionando
            self.running = True
            self.auto_mode = True
            
            # Inicia o processo de AutoTune
            asyncio.create_task(self.run())
            
        except Exception as e:
            logging.error(f"Erro no on_start: {str(e)}")
            await self.stop()
            return

    async def on_stop(self):
        try:
            # Primeiro desliga o aquecedor
            if self.heater:
                await self.actor_off(self.heater)

            # Atualiza os estados
            self.running = False
            self.auto_mode = False
            
            # Força o estado do kettle para false
            if self.kettle and self.kettle.instance:
                self.kettle.instance.state = False
                await self.cbpi.kettle.update(self.id)
                
            # Notifica o usuário
            if not self.finished:
                self.cbpi.notify('PIDRIMS AutoTune', 'Processo interrompido manualmente.', NotificationType.INFO)

        except Exception as e:
            logging.error(f"Erro no on_stop: {str(e)}")
        finally:
            # Garante que o aquecedor está desligado
            if self.heater:
                await self.actor_off(self.heater)

    async def check_pump_status(self):
        if not self.pump:
            return False
        try:
            pump_power = self.cbpi.actor.find_by_id(self.pump).power
            if pump_power <= 0:
                self.cbpi.notify('PIDRIMS AutoTune', 'Bomba parou! Interrompendo processo.', NotificationType.ERROR)
                await self.stop()
                return False
            return True
        except Exception as e:
            logging.error(f"Erro ao verificar bomba: {str(e)}")
            return False

    async def stop(self):
        try:
            # Primeiro desliga o aquecedor
            if self.heater:
                await self.actor_off(self.heater)

            # Atualiza estados
            self.running = False
            self.auto_mode = False
            
            # Força o estado do kettle para false
            if self.kettle and self.kettle.instance:
                self.kettle.instance.state = False
                await self.cbpi.kettle.update(self.id)

        except Exception as e:
            logging.error(f"Erro ao parar: {str(e)}")
        finally:
            # Garante que o aquecedor está desligado
            if self.heater:
                await self.actor_off(self.heater)

    async def check_auto_mode(self):
        try:
            if not self.kettle:
                self.kettle = self.get_kettle(self.id)

            if not self.kettle.instance.state and self.auto_mode:
                # Auto Mode foi desligado
                self.auto_mode = False
                await self.stop()
                return False
            elif self.kettle.instance.state and not self.auto_mode:
                # Auto Mode foi ligado
                await self.on_start()
            
            return self.auto_mode

        except Exception as e:
            logging.error(f"Erro ao verificar Auto Mode: {str(e)}")
            return False

    async def run(self):
        # Main execution loop for the autotuning process
        try:
            # Initialize system parameters and get current state
            self.kettle = self.get_kettle(self.id)
            self.heater = self.kettle.heater
            self.TEMP_UNIT = self.get_config_value("TEMP_UNIT", "C")
            fixedtarget = 67 if self.TEMP_UNIT == "C" else 153  # Default target if none is set
            setpoint = int(self.get_kettle_target_temp(self.id))
            current_value = self.get_sensor_value(self.kettle.sensor).get("value")

            # Log initial values for debugging
            logging.info(f"Iniciando AutoTune com os seguintes valores:")
            logging.info(f"Setpoint: {setpoint}°{self.TEMP_UNIT}")
            logging.info(f"Temperatura atual: {current_value}°{self.TEMP_UNIT}")
            logging.info(f"Output Step: {float(self.props.get('Output_Step', 100))}")
            logging.info(f"Sample Time: {self.sample_time}")
            logging.info(f"Lookback Seconds: {float(self.props.get('lookback_seconds', 30))}")

            # Verify RIMS sensor configuration
            if not self.rims_sensor:
                self.cbpi.notify('PIDRIMS AutoTune', 'Sensor RIMS não configurado!', NotificationType.ERROR)
                await self.stop()
                return

            # Verify pump configuration
            if not self.pump:
                self.cbpi.notify('PIDRIMS AutoTune', 'Bomba não configurada!', NotificationType.ERROR)
                await self.stop()
                return

            # Verify pump is running
            if not await self.check_pump_status():
                return

            # Initial safety checks and target temperature validation
            if setpoint == 0:
                if fixedtarget < current_value:
                    self.cbpi.notify('PIDRIMS AutoTune', f'Temperatura alvo não definida e temperatura atual está acima do valor padrão de {fixedtarget}°{self.TEMP_UNIT}. Por favor, defina uma temperatura alvo acima da atual ou aguarde o sistema esfriar.', NotificationType.ERROR)
                    await self.stop()
                    return
                self.cbpi.notify('PIDRIMS AutoTune', f'Temperatura alvo não definida. Sistema usará {fixedtarget}°{self.TEMP_UNIT} como alvo', NotificationType.WARNING)
                setpoint = fixedtarget
                await self.set_target_temp(self.id,setpoint)
        
            if setpoint < current_value:
                self.cbpi.notify('PIDRIMS AutoTune', 'Temperatura alvo está abaixo da temperatura atual. Escolha um setpoint mais alto ou aguarde a temperatura baixar e reinicie o AutoTune', NotificationType.ERROR)
                await self.stop()
                return

            self.cbpi.notify('PIDRIMS AutoTune', 'AutoTune em Progresso. Não desligue o modo Auto até o AutoTune terminar', NotificationType.INFO)
            
            # Initialize AutoTune parameters
            outstep = float(self.props.get("Output_Step", 100))
            outmax = float(self.props.get("Max_Output", 100))
            lookbackSec = float(self.props.get("lookback_seconds", 30))
            
            # Validate parameters
            if outstep <= 0 or outmax <= 0 or lookbackSec <= 0:
                raise ValueError(f"Parâmetros inválidos: Output Step={outstep}, Max Output={outmax}, Lookback Seconds={lookbackSec}")

            heat_percent_old = 0
            high_temp_diff_time = 0
            last_temp_check = time()

            # Create and initialize AutoTuner with more detailed error handling
            try:
                atune = AutoTuner(
                    setpoint=setpoint,
                    outputstep=outstep,
                    sampleTimeSec=self.sample_time,
                    lookbackSec=lookbackSec,
                    outputMin=0,
                    outputMax=outmax,
                    noiseband=0.5
                )
                logging.info("AutoTuner inicializado com sucesso")
            except Exception as e:
                logging.error(f"Erro ao inicializar AutoTuner: {str(e)}")
                self.cbpi.notify('PIDRIMS AutoTune', f'Erro ao inicializar AutoTune: {str(e)}', NotificationType.ERROR)
                await self.stop()
                return

            atune.log("PIDRIMS AutoTune iniciando")

            try:
                # Main autotuning loop
                await self.actor_on(self.heater, heat_percent_old)  # Ensure RIMS heater is on
                while self.running and not atune.run(self.get_sensor_value(self.kettle.sensor).get("value")):
                    # Verifica o estado do Auto Mode
                    if not await self.check_auto_mode():
                        return

                    # Check pump status first
                    if not await self.check_pump_status():
                        return

                    heat_percent = atune.output
                    current_time = time()
                    
                    # RIMS temperature safety monitoring
                    rims_temp = self.get_sensor_value(self.rims_sensor).get("value")
                    mlt_temp = self.get_sensor_value(self.kettle.sensor).get("value")
                    temp_diff = rims_temp - setpoint
                    time_elapsed = current_time - last_temp_check

                    # Log temperatures and state for debugging
                    logging.info(f"Estado atual - RIMS: {rims_temp}°{self.TEMP_UNIT}, MLT: {mlt_temp}°{self.TEMP_UNIT}, Setpoint: {setpoint}°{self.TEMP_UNIT}")
                    logging.info(f"Potência: {heat_percent}%, Estado AutoTune: {atune.state}, Picos detectados: {atune._peakCount}")
                    
                    # Safety cutoff: stop heating if RIMS temperature exceeds limit
                    if rims_temp > (setpoint + self.max_temp_diff):
                        heat_percent = 0
                        high_temp_diff_time += time_elapsed
                        
                        # Log high temperature event
                        logging.warning(f"Temperatura RIMS muito alta: {rims_temp}°{self.TEMP_UNIT}, Tempo em alta: {high_temp_diff_time:.1f}s")
                        
                        # Dynamic adjustment of AutoTune parameters based on thermal behavior
                        if high_temp_diff_time > 30:  # After 30 seconds at high temperature
                            atune._noiseband = min(2.0, atune._noiseband * 1.1)  # Increase noise tolerance
                            atune._outputstep = max(20, atune._outputstep * 0.9)  # Reduce power steps
                            logging.info(f"Ajustando parâmetros - Noise Band: {atune._noiseband}, Output Step: {atune._outputstep}")
                        
                        atune.log(f'Temperatura RIMS muito alta ({rims_temp}°{self.TEMP_UNIT}). Aquecimento desligado. Tempo em alta: {high_temp_diff_time:.1f}s')
                        self.cbpi.notify('PIDRIMS AutoTune', f'Temperatura RIMS muito alta ({rims_temp}°{self.TEMP_UNIT}). Aquecimento desligado.', NotificationType.WARNING)

                        # Force power update
                        if heat_percent_old != 0:
                            await self.actor_set_power(self.heater, 0)
                            heat_percent_old = 0
                            logging.info("Aquecedor forçado para 0%")
                    else:
                        # Reset high temperature timer when temperature is normal
                        high_temp_diff_time = max(0, high_temp_diff_time - time_elapsed)

                    # Update heater power if changed
                    if heat_percent != heat_percent_old:
                        await self.actor_set_power(self.heater, heat_percent)
                        heat_percent_old = heat_percent
                        status = "acima" if temp_diff > 0 else "abaixo"
                        logging.info(f"Potência atualizada: {heat_percent}%, RIMS {abs(temp_diff)}°{self.TEMP_UNIT} {status} do alvo")
                        atune.log(f'Potência: {heat_percent}%, RIMS {abs(temp_diff)}°{self.TEMP_UNIT} {status} do alvo, Tempo em alta: {high_temp_diff_time:.1f}s')
                    
                    last_temp_check = current_time
                    await asyncio.sleep(self.sample_time)

                # Process AutoTune results
                if atune.state == atune.STATE_SUCCEEDED:
                    logging.info("AutoTune completado com sucesso!")
                    atune.log("AutoTune completado com sucesso")
                    atune.log(f"Tempo total em temperatura alta: {high_temp_diff_time:.1f}s")
                    
                    # Calculate and log PID parameters for each tuning rule
                    for rule in atune.tuningRules:
                        params = atune.getPIDParameters(rule)
                        if high_temp_diff_time > 60:
                            params = self.adjust_params_for_temp_behavior(params, high_temp_diff_time)
                        logging.info(f"Regra: {rule} - P: {params.Kp:.8f}, I: {params.Ki:.8f}, D: {params.Kd:.8f}")
                        atune.log(f'Regra: {rule}')
                        atune.log(f'P: {params.Kp}')
                        atune.log(f'I: {params.Ki}')
                        atune.log(f'D: {params.Kd}')
                        
                        # Use RIMS-specific moderate tuning rule
                        if rule == "rims-moderate":
                            self.cbpi.notify('AutoTune completado com sucesso',
                                f"P: {params.Kp:.8f} | I: {params.Ki:.8f} | D: {params.Kd:.8f}",
                                action=[NotificationAction("OK")])
                else:
                    logging.error(f"AutoTune falhou. Estado final: {atune.state}")
                    atune.log("AutoTune falhou")
                    self.cbpi.notify('PIDRIMS AutoTune', "AutoTune falhou. Verifique os logs para mais detalhes.", action=[NotificationAction("OK")])

            except asyncio.CancelledError:
                logging.info("AutoTune cancelado pelo usuário")
                raise
            except Exception as e:
                logging.error(f"Erro durante o AutoTune: {str(e)}")
                self.cbpi.notify('PIDRIMS AutoTune', f"Erro durante o AutoTune: {str(e)}", NotificationType.ERROR)
            finally:
                await self.stop()

        except Exception as e:
            logging.error(f"Erro no método run: {str(e)}")
            self.cbpi.notify('PIDRIMS AutoTune', f"Erro no AutoTune: {str(e)}", NotificationType.ERROR)
            await self.stop()

    def adjust_params_for_temp_behavior(self, params, high_temp_time):
        # Adjusts PID parameters based on observed thermal behavior
        #
        # Args:
        #     params: Current PID parameters
        #     high_temp_time: Time spent at high temperatures
        #            
        # Returns:
        #     Adjusted PID parameters optimized for RIMS safety
        
        # Calculate adjustment factor based on time spent at high temperatures
        adjustment_factor = min(1.5, 1 + (high_temp_time / 300))  # Maximum 50% adjustment
        
        # Adjust each parameter for better temperature control:
        # - Increase P for faster response to temperature changes
        # - Reduce I to minimize overshoot
        # - Increase D for better overshoot prevention
        Kp = params.Kp * adjustment_factor
        Ki = params.Ki / adjustment_factor
        Kd = params.Kd * adjustment_factor
        
        return self.PIDParams(Kp=Kp, Ki=Ki, Kd=Kd)

# Based on a fork of Arduino PID AutoTune Library
# See https://github.com/t0mpr1c3/Arduino-PID-AutoTune-Library
class AutoTuner(object):
	PIDParams = namedtuple('PIDParams', ['Kp', 'Ki', 'Kd'])

	PEAK_AMPLITUDE_TOLERANCE = 0.05
	STATE_OFF = 'off'
	STATE_RELAY_STEP_UP = 'relay step up'
	STATE_RELAY_STEP_DOWN = 'relay step down'
	STATE_SUCCEEDED = 'succeeded'
	STATE_FAILED = 'failed'

	_tuning_rules = {
		# rule: [Kp_divisor, Ki_divisor, Kd_divisor]
		"ziegler-nichols": [34, 40, 160],
		"tyreus-luyben": [44,  9, 126],
		"ciancone-marlin": [66, 88, 162],
		"pessen-integral": [28, 50, 133],
		"some-overshoot": [60, 40,  60],
		"no-overshoot": [100, 40,  60],
		"brewing": [2.5, 3, 3600],
		"rims-aggressive": [2.0, 2.5, 3000],    # Mais agressivo para sistemas RIMS
		"rims-moderate": [3.0, 3.5, 4000],      # Moderado para sistemas RIMS
		"rims-conservative": [4.0, 4.5, 5000]    # Conservador para sistemas RIMS
	}

	def __init__(self, setpoint, outputstep=10, sampleTimeSec=5, lookbackSec=60,
				 outputMin=float('-inf'), outputMax=float('inf'), noiseband=0.5, getTimeMs=None):
		if setpoint is None or setpoint <= 0:
			raise ValueError('Setpoint deve ser especificado e maior que zero')
		if outputstep < 1:
			raise ValueError('Output step % deve ser maior ou igual a 1')
		if sampleTimeSec < 1:
			raise ValueError('Sample Time Seconds deve ser maior ou igual a 1')
		if lookbackSec < sampleTimeSec:
			raise ValueError('Lookback Seconds deve ser maior ou igual a Sample Time Seconds (5)')
		if outputMin >= outputMax:
			raise ValueError('Min Output % deve ser menor que Max Output %')
		if noiseband <= 0:
			raise ValueError('Noise Band deve ser maior que zero')

		try:
			self._inputs = deque(maxlen=round(lookbackSec / sampleTimeSec))
			self._sampleTime = sampleTimeSec * 1000
			self._setpoint = float(setpoint)  # Garante que setpoint é float
			self._outputstep = float(outputstep)  # Garante que outputstep é float
			self._noiseband = float(noiseband)  # Garante que noiseband é float
			self._outputMin = float(outputMin)  # Garante que outputMin é float
			self._outputMax = float(outputMax)  # Garante que outputMax é float
		except ValueError as e:
			raise ValueError(f'Erro ao converter parâmetros para float: {str(e)}')
		except Exception as e:
			raise ValueError(f'Erro ao inicializar parâmetros: {str(e)}')

		self._state = AutoTuner.STATE_OFF
		self._peakTimestamps = deque(maxlen=5)
		self._peaks = deque(maxlen=5)

		self._output = 0
		self._lastRunTimestamp = 0
		self._peakType = 0
		self._peakCount = 0
		self._initialOutput = 0
		self._inducedAmplitude = 0
		self._Ku = 0
		self._Pu = 0

		if getTimeMs is None:
			self._getTimeMs = self._currentTimeMs
		else:
			self._getTimeMs = getTimeMs

		# Log da inicialização
		logging.info(f"AutoTuner inicializado - Setpoint: {self._setpoint}, Output Step: {self._outputstep}, Sample Time: {sampleTimeSec}s, Lookback: {lookbackSec}s")

	@property
	def state(self):
		return self._state

	@property
	def output(self):
		return self._output

	@property
	def tuningRules(self):
		return self._tuning_rules.keys()

	def getPIDParameters(self, tuningRule='ziegler-nichols'):
		divisors = self._tuning_rules[tuningRule]
		kp = self._Ku / divisors[0]
		ki = kp / (self._Pu / divisors[1])
		kd = kp * (self._Pu / divisors[2])
		return AutoTuner.PIDParams(kp, ki, kd)

	def log(self, text):
		filename = "./logs/autotune.log"
		formatted_time = strftime("%Y-%m-%d %H:%M:%S", localtime())

		with open(filename, "a") as file:
			file.write("%s,%s\n" % (formatted_time, text))
		
	def run(self, inputValue):
		now = self._getTimeMs()

		if (self._state == AutoTuner.STATE_OFF
				or self._state == AutoTuner.STATE_SUCCEEDED
				or self._state == AutoTuner.STATE_FAILED):
			self._initTuner(inputValue, now)
		elif (now - self._lastRunTimestamp) < self._sampleTime:
			return False

		self._lastRunTimestamp = now

		# check input and change relay state if necessary
		if (self._state == AutoTuner.STATE_RELAY_STEP_UP
				and inputValue > self._setpoint + self._noiseband):
			self._state = AutoTuner.STATE_RELAY_STEP_DOWN
			self.log('switched state: {0}'.format(self._state))
			self.log('input: {0}'.format(inputValue))
		elif (self._state == AutoTuner.STATE_RELAY_STEP_DOWN
				and inputValue < self._setpoint - self._noiseband):
			self._state = AutoTuner.STATE_RELAY_STEP_UP
			self.log('switched state: {0}'.format(self._state))
			self.log('input: {0}'.format(inputValue))

		# set output
		if (self._state == AutoTuner.STATE_RELAY_STEP_UP):
			self._output = self._initialOutput + self._outputstep
		elif self._state == AutoTuner.STATE_RELAY_STEP_DOWN:
			self._output = self._initialOutput - self._outputstep

		# respect output limits
		self._output = min(self._output, self._outputMax)
		self._output = max(self._output, self._outputMin)

		# identify peaks
		isMax = True
		isMin = True

		for val in self._inputs:
			isMax = isMax and (inputValue > val)
			isMin = isMin and (inputValue < val)

		self._inputs.append(inputValue)

		# we don't want to trust the maxes or mins until the input array is full
		if len(self._inputs) < self._inputs.maxlen:
			return False

		# increment peak count and record peak time for maxima and minima
		inflection = False

		# peak types:
		# -1: minimum
		# +1: maximum
		if isMax:
			if self._peakType == -1:
				inflection = True
			self._peakType = 1
		elif isMin:
			if self._peakType == 1:
				inflection = True
			self._peakType = -1

		# update peak times and values
		if inflection:
			self._peakCount += 1
			self._peaks.append(inputValue)
			self._peakTimestamps.append(now)
			self.log('found peak: {0}'.format(inputValue))
			self.log('peak count: {0}'.format(self._peakCount))

		# check for convergence of induced oscillation
		# convergence of amplitude assessed on last 4 peaks (1.5 cycles)
		self._inducedAmplitude = 0

		if inflection and (self._peakCount > 4):
			absMax = self._peaks[-2]
			absMin = self._peaks[-2]
			for i in range(0, len(self._peaks) - 2):
				self._inducedAmplitude += abs(self._peaks[i] - self._peaks[i+1])
				absMax = max(self._peaks[i], absMax)
				absMin = min(self._peaks[i], absMin)

			self._inducedAmplitude /= 6.0

			# check convergence criterion for amplitude of induced oscillation
			amplitudeDev = ((0.5 * (absMax - absMin) - self._inducedAmplitude)
							/ self._inducedAmplitude)

			self.log('amplitude: {0}'.format(self._inducedAmplitude))
			self.log('amplitude deviation: {0}'.format(amplitudeDev))

			if amplitudeDev < AutoTuner.PEAK_AMPLITUDE_TOLERANCE:
				self._state = AutoTuner.STATE_SUCCEEDED

		# if the autotune has not already converged
		# terminate after 10 cycles
		if self._peakCount >= 20:
			self._output = 0
			self._state = AutoTuner.STATE_FAILED
			return True

		if self._state == AutoTuner.STATE_SUCCEEDED:
			self._output = 0

			# calculate ultimate gain
			self._Ku = 4.0 * self._outputstep / (self._inducedAmplitude * math.pi)

			# calculate ultimate period in seconds
			period1 = self._peakTimestamps[3] - self._peakTimestamps[1]
			period2 = self._peakTimestamps[4] - self._peakTimestamps[2]
			self._Pu = 0.5 * (period1 + period2) / 1000.0
			return True

		return False

	def _currentTimeMs(self):
		return time.time() * 1000

	def _initTuner(self, inputValue, timestamp):
		self._peakType = 0
		self._peakCount = 0
		self._output = 0
		self._initialOutput = 0
		self._Ku = 0
		self._Pu = 0
		self._inputs.clear()
		self._peaks.clear()
		self._peakTimestamps.clear()
		self._peakTimestamps.append(timestamp)
		self._state = AutoTuner.STATE_RELAY_STEP_UP

def setup(cbpi):
    '''
    This method is called by the server during startup 
    Here you need to register your plugins at the server
    
    :param cbpi: the cbpi core 
    :return: 
    '''
    # Register the plugin with the correct name
    cbpi.plugin.register("PIDRIMSAutoTune", PIDRIMSAutotune)
    # Register the plugin as a kettle logic type
    #cbpi.kettle.types["PIDRIMSAutoTune"] = PIDRIMSAutotune
