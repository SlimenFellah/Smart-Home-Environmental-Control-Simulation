"""
Smart Home Environmental Control Simulation
==========================================

This simulation models intelligent agents that regulate temperature, luminosity,
and air quality in a smart home environment. The agents use reinforcement learning
to optimize comfort while minimizing energy usage.

Date: February 28, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from collections import deque
import random
import time
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# Set random seed for reproducibility
np.random.seed(42)

class Environment:
    """
    Simulates the physical environment of a smart home, including
    temperature, luminosity, and air quality.
    """
    
    def __init__(self, rooms=4):
        self.rooms = rooms
        
        # Environmental variables for each room
        self.temperature = np.ones(rooms) * 22.0  # Initial temperature in Celsius
        self.target_temperature = np.ones(rooms) * 22.0
        self.luminosity = np.ones(rooms) * 300.0  # Initial luminosity in lux
        self.target_luminosity = np.ones(rooms) * 300.0
        self.air_quality = np.ones(rooms) * 400.0  # Initial CO2 level in ppm
        self.target_air_quality = np.ones(rooms) * 400.0
        
        # Device states
        self.heater_state = np.zeros(rooms)  # 0=off, 1=on
        self.ac_state = np.zeros(rooms)      # 0=off, 1=on
        self.light_state = np.zeros(rooms)   # 0-1.0 (dimmer)
        self.ventilation_state = np.zeros(rooms)  # 0-1.0 (fan speed)
        
        # Outdoor conditions
        self.outdoor_temperature = 15.0
        self.outdoor_luminosity = 10000.0  # Bright day
        self.outdoor_air_quality = 350.0  # Good air quality
        
        # Time simulation
        self.time = datetime(2025, 2, 28, 8, 0)  # Start at 8 AM
        self.time_step = timedelta(minutes=10)  # 10-minute intervals
        
        # Room occupancy (0=empty, 1=occupied)
        self.occupancy = np.zeros(rooms)
        
        # Energy consumption
        self.energy_consumption = 0.0
        self.energy_history = []
        
        # Comfort history (100 = perfect comfort)
        self.comfort_history = []
        
        # Data history for visualization
        self.history = {
            'time': [],
            'temperature': [[] for _ in range(rooms)],
            'luminosity': [[] for _ in range(rooms)],
            'air_quality': [[] for _ in range(rooms)],
            'occupancy': [[] for _ in range(rooms)],
            'energy': []
        }
    
    def step(self):
        """Advance the environment by one time step"""
        # Update time
        self.time += self.time_step
        
        # Update outdoor conditions based on time of day
        hour = self.time.hour + self.time.minute / 60.0
        self.outdoor_temperature = 15.0 + 5.0 * np.sin(np.pi * (hour - 6) / 12)  # Peak at noon
        self.outdoor_luminosity = max(0, 15000 * np.sin(np.pi * (hour - 6) / 12))  # Daylight hours
        
        # Random fluctuations
        self.outdoor_temperature += np.random.normal(0, 0.5)
        self.outdoor_luminosity += np.random.normal(0, 500)
        self.outdoor_air_quality += np.random.normal(0, 5)
        
        # Update room occupancy based on time
        self.update_occupancy(hour)
        
        # Apply device effects to environment
        self.apply_device_effects()
        
        # Natural environment changes
        self.apply_natural_changes()
        
        # Calculate energy consumption
        self.calculate_energy()
        
        # Calculate comfort level
        comfort = self.calculate_comfort()
        self.comfort_history.append(comfort)
        
        # Store history for visualization
        self.history['time'].append(self.time)
        for i in range(self.rooms):
            self.history['temperature'][i].append(self.temperature[i])
            self.history['luminosity'][i].append(self.luminosity[i])
            self.history['air_quality'][i].append(self.air_quality[i])
            self.history['occupancy'][i].append(self.occupancy[i])
        self.history['energy'].append(self.energy_consumption)
        
        # Return the current state of the environment
        return self.get_state()
    
    def update_occupancy(self, hour):
        """Update room occupancy based on time of day"""
        # Probabilities of occupancy for different rooms at different times
        # Simple model: people wake up, go to kitchen/living room during day, return to bedroom at night
        if 7 <= hour < 9:  # Morning time
            self.occupancy[0] = np.random.choice([0, 1], p=[0.1, 0.9])  # Bedroom
            self.occupancy[1] = np.random.choice([0, 1], p=[0.6, 0.4])  # Living room
            self.occupancy[2] = np.random.choice([0, 1], p=[0.3, 0.7])  # Kitchen
            self.occupancy[3] = np.random.choice([0, 1], p=[0.8, 0.2])  # Bathroom
        elif 9 <= hour < 17:  # Daytime
            self.occupancy[0] = np.random.choice([0, 1], p=[0.9, 0.1])  # Bedroom
            self.occupancy[1] = np.random.choice([0, 1], p=[0.4, 0.6])  # Living room
            self.occupancy[2] = np.random.choice([0, 1], p=[0.7, 0.3])  # Kitchen
            self.occupancy[3] = np.random.choice([0, 1], p=[0.8, 0.2])  # Bathroom
        elif 17 <= hour < 23:  # Evening
            self.occupancy[0] = np.random.choice([0, 1], p=[0.7, 0.3])  # Bedroom
            self.occupancy[1] = np.random.choice([0, 1], p=[0.3, 0.7])  # Living room
            self.occupancy[2] = np.random.choice([0, 1], p=[0.6, 0.4])  # Kitchen
            self.occupancy[3] = np.random.choice([0, 1], p=[0.7, 0.3])  # Bathroom
        else:  # Night time
            self.occupancy[0] = np.random.choice([0, 1], p=[0.1, 0.9])  # Bedroom
            self.occupancy[1] = np.random.choice([0, 1], p=[0.9, 0.1])  # Living room
            self.occupancy[2] = np.random.choice([0, 1], p=[0.9, 0.1])  # Kitchen
            self.occupancy[3] = np.random.choice([0, 1], p=[0.8, 0.2])  # Bathroom
        
        # Update target values based on occupancy
        for i in range(self.rooms):
            if self.occupancy[i] > 0:
                # Rooms are occupied, set comfort targets
                self.target_temperature[i] = 22.0
                self.target_luminosity[i] = 300.0
                self.target_air_quality[i] = 400.0
            else:
                # Rooms are empty, set eco targets
                self.target_temperature[i] = 19.0 if self.outdoor_temperature < 15 else 25.0
                self.target_luminosity[i] = 0.0
                self.target_air_quality[i] = 600.0
    
    def apply_device_effects(self):
        """Apply the effects of devices on the environment"""
        for i in range(self.rooms):
            # Temperature effects
            if self.heater_state[i] > 0:
                self.temperature[i] += 0.5 * self.heater_state[i]
            if self.ac_state[i] > 0:
                self.temperature[i] -= 0.5 * self.ac_state[i]
            
            # Luminosity effects
            # Scale based on natural light availability
            natural_light = max(0, self.outdoor_luminosity * 0.1)  # 10% of outdoor light gets in
            self.luminosity[i] = natural_light + self.light_state[i] * 500
            
            # Air quality effects
            # Occupancy worsens air quality, ventilation improves it
            if self.occupancy[i] > 0:
                self.air_quality[i] += 10  # CO2 increase from occupants
            
            # Ventilation brings indoor air quality toward outdoor
            ventilation_effect = self.ventilation_state[i] * 0.1
            self.air_quality[i] = (1 - ventilation_effect) * self.air_quality[i] + ventilation_effect * self.outdoor_air_quality
    
    def apply_natural_changes(self):
        """Apply natural environmental changes over time"""
        for i in range(self.rooms):
            # Temperature naturally moves toward outdoor temperature
            temp_diff = self.outdoor_temperature - self.temperature[i]
            self.temperature[i] += 0.02 * temp_diff
            
            # Add small random fluctuations
            self.temperature[i] += np.random.normal(0, 0.1)
            self.luminosity[i] += np.random.normal(0, 10)
            self.air_quality[i] += np.random.normal(0, 2)
            
            # Ensure values stay in reasonable ranges
            self.temperature[i] = max(10, min(35, self.temperature[i]))
            self.luminosity[i] = max(0, min(20000, self.luminosity[i]))
            self.air_quality[i] = max(300, min(1500, self.air_quality[i]))
    
    def calculate_energy(self):
        """Calculate energy consumption for this time step"""
        # Reset energy counter for this step
        self.energy_consumption = 0
        
        # Calculate energy from all devices
        for i in range(self.rooms):
            # Heating (kW)
            self.energy_consumption += self.heater_state[i] * 2.0
            
            # AC (kW)
            self.energy_consumption += self.ac_state[i] * 1.5
            
            # Lighting (kW)
            self.energy_consumption += self.light_state[i] * 0.1
            
            # Ventilation (kW)
            self.energy_consumption += self.ventilation_state[i] * 0.2
        
        # Convert to kWh for this time step (10 minutes = 1/6 hour)
        energy_kwh = self.energy_consumption * (1/6)
        self.energy_history.append(energy_kwh)
    
    def calculate_comfort(self):
        """Calculate comfort level across all occupied rooms (0-100)"""
        total_comfort = 0
        occupied_rooms = 0
        
        for i in range(self.rooms):
            if self.occupancy[i] > 0:
                occupied_rooms += 1
                room_comfort = 0
                
                # Temperature comfort (ideal: target_temperature ± 1°C)
                temp_diff = abs(self.temperature[i] - self.target_temperature[i])
                if temp_diff <= 1.0:
                    temp_comfort = 100
                else:
                    temp_comfort = max(0, 100 - (temp_diff - 1) * 20)
                
                # Luminosity comfort (ideal: target_luminosity ± 50 lux)
                lum_diff = abs(self.luminosity[i] - self.target_luminosity[i])
                if lum_diff <= 50:
                    lum_comfort = 100
                else:
                    lum_comfort = max(0, 100 - (lum_diff - 50) * 0.2)
                
                # Air quality comfort (ideal: target_air_quality ± 50 ppm)
                aq_diff = abs(self.air_quality[i] - self.target_air_quality[i])
                if aq_diff <= 50:
                    aq_comfort = 100
                else:
                    aq_comfort = max(0, 100 - (aq_diff - 50) * 0.2)
                
                # Overall room comfort (equal weighting)
                room_comfort = (temp_comfort + lum_comfort + aq_comfort) / 3
                total_comfort += room_comfort
        
        # Return average comfort for occupied rooms
        if occupied_rooms > 0:
            return total_comfort / occupied_rooms
        else:
            return 100  # If no occupied rooms, comfort is perfect (no need to maintain comfort)
    
    def get_state(self):
        """Get the current state of the environment for each room"""
        state = {
            'time': self.time,
            'outdoor_temperature': self.outdoor_temperature,
            'outdoor_luminosity': self.outdoor_luminosity,
            'outdoor_air_quality': self.outdoor_air_quality,
            'rooms': []
        }
        
        for i in range(self.rooms):
            room_state = {
                'temperature': self.temperature[i],
                'target_temperature': self.target_temperature[i],
                'luminosity': self.luminosity[i],
                'target_luminosity': self.target_luminosity[i],
                'air_quality': self.air_quality[i],
                'target_air_quality': self.target_air_quality[i],
                'occupancy': self.occupancy[i],
                'heater': self.heater_state[i],
                'ac': self.ac_state[i],
                'light': self.light_state[i],
                'ventilation': self.ventilation_state[i]
            }
            state['rooms'].append(room_state)
        
        return state
    
    def set_device_states(self, room, device_type, value):
        """Set the state of a device in a room"""
        if 0 <= room < self.rooms:
            if device_type == 'heater':
                self.heater_state[room] = max(0, min(1, value))
                # Turn off AC if heater is on
                if value > 0:
                    self.ac_state[room] = 0
            elif device_type == 'ac':
                self.ac_state[room] = max(0, min(1, value))
                # Turn off heater if AC is on
                if value > 0:
                    self.heater_state[room] = 0
            elif device_type == 'light':
                self.light_state[room] = max(0, min(1, value))
            elif device_type == 'ventilation':
                self.ventilation_state[room] = max(0, min(1, value))


class IntelligentAgent:
    """
    Intelligent agent that controls environmental parameters in a smart home.
    Uses a Q-learning approach to optimize comfort and energy usage.
    """
    
    def __init__(self, room_id, environment):
        self.room_id = room_id
        self.environment = environment
        
        # Device control ranges
        self.control_ranges = {
            'heater': [0, 0.5, 1.0],
            'ac': [0, 0.5, 1.0],
            'light': [0, 0.25, 0.5, 0.75, 1.0],
            'ventilation': [0, 0.25, 0.5, 0.75, 1.0]
        }
        
        # Q-learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.2
        self.exploration_decay = 0.995
        self.min_exploration_rate = 0.01
        
        # Initialize Q-table - we'll discretize the state space
        self.q_table = {}
        
        # Performance metrics
        self.reward_history = []
        self.action_history = []
    
    def discretize_state(self, room_state):
        """Convert continuous state to discrete state for Q-learning"""
        # Get the current temperature, luminosity, air quality, and occupancy
        temp = room_state['temperature']
        target_temp = room_state['target_temperature']
        lum = room_state['luminosity']
        target_lum = room_state['target_luminosity']
        aq = room_state['air_quality']
        target_aq = room_state['target_air_quality']
        occupancy = room_state['occupancy']
        
        # Discretize the temperature difference
        temp_diff = temp - target_temp
        if temp_diff < -2:
            temp_state = "too_cold"
        elif temp_diff < -0.5:
            temp_state = "cold"
        elif temp_diff < 0.5:
            temp_state = "optimal"
        elif temp_diff < 2:
            temp_state = "warm"
        else:
            temp_state = "too_warm"
        
        # Discretize the luminosity difference
        lum_diff = lum - target_lum
        if target_lum == 0:  # No light needed
            lum_state = "off" if lum < 50 else "too_bright"
        elif lum_diff < -100:
            lum_state = "too_dark"
        elif lum_diff < -30:
            lum_state = "dark"
        elif lum_diff < 30:
            lum_state = "optimal"
        elif lum_diff < 100:
            lum_state = "bright"
        else:
            lum_state = "too_bright"
        
        # Discretize the air quality difference
        aq_diff = aq - target_aq
        if aq_diff < -50:
            aq_state = "very_good"
        elif aq_diff < 0:
            aq_state = "good"
        elif aq_diff < 50:
            aq_state = "acceptable"
        elif aq_diff < 200:
            aq_state = "poor"
        else:
            aq_state = "very_poor"
        
        # Return the discretized state as a tuple (for dictionary key)
        return (temp_state, lum_state, aq_state, occupancy)
    
    def get_actions(self):
        """Get all possible device control combinations"""
        actions = []
        
        # If room is unoccupied, we have a simpler action space (energy saving)
        current_state = self.environment.get_state()['rooms'][self.room_id]
        if current_state['occupancy'] == 0:
            # Just offer eco-mode options
            actions.append({
                'heater': 0,
                'ac': 0,
                'light': 0,
                'ventilation': 0.25  # Minimal ventilation
            })
            return actions
        
        # Generate all combinations (this could be optimized to reduce combinations)
        for heater in self.control_ranges['heater']:
            for ac in self.control_ranges['ac']:
                # Skip if both heater and AC are on
                if heater > 0 and ac > 0:
                    continue
                    
                for light in self.control_ranges['light']:
                    for ventilation in self.control_ranges['ventilation']:
                        action = {
                            'heater': heater,
                            'ac': ac,
                            'light': light,
                            'ventilation': ventilation
                        }
                        actions.append(action)
        
        return actions
    
    def get_q_value(self, state, action):
        """Get Q-value for a state-action pair, initializing if necessary"""
        # Convert action dictionary to a hashable tuple
        action_tuple = tuple(action.items())
        
        # Create state-action key
        key = (state, action_tuple)
        
        # Return existing or initialize new Q-value
        if key not in self.q_table:
            self.q_table[key] = 0.0
        
        return self.q_table[key]
    
    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using Q-learning formula"""
        # Get the best next action
        next_actions = self.get_actions()
        next_q_values = [self.get_q_value(next_state, a) for a in next_actions]
        max_next_q = max(next_q_values) if next_q_values else 0
        
        # Convert action to tuple for dictionary key
        action_tuple = tuple(action.items())
        key = (state, action_tuple)
        
        # Q-learning update
        current_q = self.q_table.get(key, 0.0)
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[key] = new_q
    
    def choose_action(self, state):
        """Choose an action using epsilon-greedy policy"""
        # Get all possible actions
        actions = self.get_actions()
        
        # Exploration: choose random action
        if np.random.random() < self.exploration_rate:
            action = random.choice(actions)
            return action
        
        # Exploitation: choose best action
        q_values = [self.get_q_value(state, a) for a in actions]
        max_q = max(q_values)
        
        # Find all actions with the max Q-value
        best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        
        # Choose randomly among best actions
        action = random.choice(best_actions)
        return action
    
    def calculate_reward(self, room_state, action, energy_consumption):
        """Calculate reward based on comfort and energy efficiency"""
        # Get environment state
        temp = room_state['temperature']
        target_temp = room_state['target_temperature']
        lum = room_state['luminosity']
        target_lum = room_state['target_luminosity']
        aq = room_state['air_quality']
        target_aq = room_state['target_air_quality']
        occupancy = room_state['occupancy']
        
        # Calculate comfort component
        temp_diff = abs(temp - target_temp)
        lum_diff = abs(lum - target_lum)
        aq_diff = abs(aq - target_aq)
        
        # Temperature comfort (high reward when close to target)
        if temp_diff < 0.5:
            temp_reward = 10
        elif temp_diff < 1.0:
            temp_reward = 5
        elif temp_diff < 2.0:
            temp_reward = 0
        else:
            temp_reward = -5 * (temp_diff - 2.0)
        
        # Luminosity comfort
        if target_lum == 0:  # Lights should be off
            lum_reward = 10 if lum < 50 else -lum / 50
        else:  # Lights should be at target
            if lum_diff < 30:
                lum_reward = 10
            elif lum_diff < 100:
                lum_reward = 5
            elif lum_diff < 200:
                lum_reward = 0
            else:
                lum_reward = -5
        
        # Air quality comfort
        if aq_diff < 50:
            aq_reward = 10
        elif aq_diff < 100:
            aq_reward = 5
        elif aq_diff < 200:
            aq_reward = 0
        else:
            aq_reward = -5
        
        # Energy consumption penalty
        energy_penalty = -2 * energy_consumption
        
        # Combine rewards with weights
        if occupancy > 0:
            # Occupied room: prioritize comfort
            total_reward = 0.5 * temp_reward + 0.3 * lum_reward + 0.2 * aq_reward + 0.1 * energy_penalty
        else:
            # Unoccupied room: prioritize energy savings
            total_reward = 0.1 * temp_reward + 0.1 * lum_reward + 0.1 * aq_reward + 0.7 * energy_penalty
        
        return total_reward
    
    def act(self):
        """Take an action in the environment"""
        # Get current state
        current_env_state = self.environment.get_state()
        room_state = current_env_state['rooms'][self.room_id]
        current_state = self.discretize_state(room_state)
        
        # Choose action
        action = self.choose_action(current_state)
        
        # Apply action to environment
        for device, value in action.items():
            self.environment.set_device_states(self.room_id, device, value)
        
        # Store action for history
        self.action_history.append(action)
        
        # Step the environment
        next_env_state = self.environment.step()
        next_room_state = next_env_state['rooms'][self.room_id]
        next_state = self.discretize_state(next_room_state)
        
        # Calculate reward
        reward = self.calculate_reward(
            next_room_state,
            action,
            self.environment.energy_consumption
        )
        
        # Update Q-value
        self.update_q_value(current_state, action, reward, next_state)
        
        # Store reward for history
        self.reward_history.append(reward)
        
        # Decay exploration rate
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )
        
        return next_env_state, reward


class Visualizer:
    """
    Provides visualization tools for the smart home environment.
    """
    
    def __init__(self, environment):
        self.environment = environment
        
        # Pre-define colors for each room
        self.room_colors = ['#ff9999', '#99ff99', '#9999ff', '#ffff99']
        self.room_names = ['Bedroom', 'Living Room', 'Kitchen', 'Bathroom']
    
    def setup_dashboard(self):
        """Set up an interactive dashboard for the simulation"""
        plt.ion()  # Interactive mode
        self.fig = plt.figure(figsize=(15, 10))
        self.fig.suptitle('Smart Home Environment Simulation', fontsize=16)
        
        # Create subplots grid
        self.ax_temp = plt.subplot2grid((3, 3), (0, 0), colspan=2)
        self.ax_lum = plt.subplot2grid((3, 3), (1, 0), colspan=2)
        self.ax_aq = plt.subplot2grid((3, 3), (2, 0), colspan=2)
        
        self.ax_energy = plt.subplot2grid((3, 3), (0, 2))
        self.ax_comfort = plt.subplot2grid((3, 3), (1, 2))
        self.ax_home = plt.subplot2grid((3, 3), (2, 2))
        
        # Initialize plots
        self.temp_lines = []
        self.lum_lines = []
        self.aq_lines = []
        
        for i in range(self.environment.rooms):
            temp_line, = self.ax_temp.plot([], [], label=self.room_names[i], color=self.room_colors[i])
            self.temp_lines.append(temp_line)
            
            lum_line, = self.ax_lum.plot([], [], label=self.room_names[i], color=self.room_colors[i])
            self.lum_lines.append(lum_line)
            
            aq_line, = self.ax_aq.plot([], [], label=self.room_names[i], color=self.room_colors[i])
            self.aq_lines.append(aq_line)
        
        # Setup temperature plot
        self.ax_temp.set_title('Temperature (°C)')
        self.ax_temp.set_xlabel('Time')
        self.ax_temp.set_ylabel('Temperature (°C)')
        self.ax_temp.grid(True)
        self.ax_temp.legend(loc='upper right')
        
        # Setup luminosity plot
        self.ax_lum.set_title('Luminosity (lux)')
        self.ax_lum.set_xlabel('Time')
        self.ax_lum.set_ylabel('Luminosity (lux)')
        self.ax_lum.grid(True)
        self.ax_lum.legend(loc='upper right')
        
        # Setup air quality plot
        self.ax_aq.set_title('Air Quality (CO2 ppm)')
        self.ax_aq.set_xlabel('Time')
        self.ax_aq.set_ylabel('CO2 (ppm)')
        self.ax_aq.grid(True)
        self.ax_aq.legend(loc='upper right')
        
        # Setup energy consumption plot
        self.energy_bar = self.ax_energy.bar(['Energy'], [0], color='orange')
        self.ax_energy.set_title('Current Energy Use (kW)')
        self.ax_energy.set_ylim(0, 10)
        
        # Setup comfort level plot
        self.comfort_bar = self.ax_comfort.bar(['Comfort'], [0], color='green')
        self.ax_comfort.set_title('Comfort Level (%)')
        self.ax_comfort.set_ylim(0, 100)
        
        # Setup house visualization
        self.draw_house()
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
    
    def draw_house(self):
        """Draw a simple house diagram showing room states"""
        self.ax_home.clear()
        self.ax_home.set_title('Home Status')
        self.ax_home.set_xlim(0, 10)
        self.ax_home.set_ylim(0, 10)
        self.ax_home.axis('off')
        
        # Draw rooms
        rooms = [
            (1, 1, 4, 4, 0),  # Bedroom
            (5, 1, 4, 4, 1),  # Living Room
            (1, 5, 4, 4, 2),  # Kitchen
            (5, 5, 4, 4, 3),  # Bathroom
        ]
        
        for x, y, w, h, room_id in rooms:
            # Get room state
            state = self.environment.get_state()
            room_state = state['rooms'][room_id]
            
            # Get color based on temperature (blue=cold, red=hot)
            temp = room_state['temperature']
            temp_norm = (temp - 15) / 15  # Normalize between 15-30°C
            temp_color = (min(1, max(0, temp_norm)), 0, min(1, max(0, 1-temp_norm)))
            
            # Draw room rectangle with temperature color
            rectangle = plt.Rectangle((x, y), w, h, 
                                     facecolor=temp_color, 
                                     alpha=0.3,
                                     edgecolor='black',
                                     linewidth=2)
            self.ax_home.add_patch(rectangle)
            
            # Add room name
            self.ax_home.text(x + w/2, y + h - 0.5, self.room_names[room_id],
                             horizontalalignment='center',
                             verticalalignment='center',
                             fontsize=9)
            
            # Show occupancy
            if room_state['occupancy'] > 0:
                self.ax_home.plot(x + w/2, y + h/2, 'o', markersize=10, color='black')
            
            # Show device states with icons
            icons = []
            # Heater
            if room_state['heater'] > 0:
                heater_icon = self.ax_home.text(x + 0.5, y + 0.5, '🔥', fontsize=12)
                icons.append(heater_icon)
            
            # AC
            if room_state['ac'] > 0:
                ac_icon = self.ax_home.text(x + 1.5, y + 0.5, '❄️', fontsize=12)
                icons.append(ac_icon)
            
            # Light
            if room_state['light'] > 0:
                light_icon = self.ax_home.text(x + 2.5, y + 0.5, '💡', fontsize=12)
                icons.append(light_icon)
            
            # Ventilation
            if room_state['ventilation'] > 0:
                vent_icon = self.ax_home.text(x + 3.5, y + 0.5, '🌬️', fontsize=12)
                icons.append(vent_icon)
            
            # Add stats text
            stats = f"T: {room_state['temperature']:.1f}°C\n" \
                   f"L: {room_state['luminosity']:.0f} lux\n" \
                   f"AQ: {room_state['air_quality']:.0f} ppm"
            self.ax_home.text(x + w/2, y + 2, stats,
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontsize=8)
    

    def update_plots(self):
        """Update all plots with the latest data"""
        # Get time data for x-axis
        times = self.environment.history['time']
        formatted_times = [t.strftime('%H:%M') for t in times]  # For labels
        numeric_times = mdates.date2num(times)  # Convert times to numerical format for plotting

        # Update temperature lines
        for i in range(self.environment.rooms):
            self.temp_lines[i].set_data(numeric_times, self.environment.history['temperature'][i])

        # Update luminosity lines
        for i in range(self.environment.rooms):
            self.lum_lines[i].set_data(numeric_times, self.environment.history['luminosity'][i])

        # Update air quality lines
        for i in range(self.environment.rooms):
            self.aq_lines[i].set_data(numeric_times, self.environment.history['air_quality'][i])

        # Update energy bar
        self.energy_bar[0].set_height(self.environment.energy_consumption)

        # Update comfort bar
        if self.environment.comfort_history:
            self.comfort_bar[0].set_height(self.environment.comfort_history[-1])

        # Update house visualization
        self.draw_house()

        # Adjust x-axis limits if needed
        if times:
            for ax in [self.ax_temp, self.ax_lum, self.ax_aq]:
                ax.set_xticks(numeric_times[::max(1, len(numeric_times)//10)])
                ax.set_xticklabels(formatted_times[::max(1, len(formatted_times)//10)], rotation=45)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # Format x-axis labels
                ax.relim()
                ax.autoscale_view()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
class SmartHomeSimulation:
    """
    Main simulation class that brings together the environment, agents, and visualization.
    """
    
    def __init__(self, rooms=4, simulation_duration=24):
        """
        Initialize the simulation.
        
        Args:
            rooms: Number of rooms in the smart home
            simulation_duration: Duration in hours to simulate
        """
        self.environment = Environment(rooms=rooms)
        self.simulation_duration = simulation_duration
        
        # Create intelligent agents for each room
        self.agents = []
        for i in range(rooms):
            agent = IntelligentAgent(i, self.environment)
            self.agents.append(agent)
        
        # Create visualizer
        self.visualizer = Visualizer(self.environment)
        
        # Results storage
        self.results = {
            'energy_total': 0,
            'comfort_avg': 0,
            'occupancy_hours': 0
        }
    
    def run(self, steps_per_hour=6, display_interval=1):
        """
        Run the simulation.
        
        Args:
            steps_per_hour: Number of simulation steps per hour
            display_interval: How often to update the visualization (in steps)
        """
        # Setup visualization
        self.visualizer.setup_dashboard()
        
        # Calculate total steps
        total_steps = self.simulation_duration * steps_per_hour
        
        # Run simulation
        for step in range(total_steps):
            # Agent actions and environment step
            for agent in self.agents:
                agent.act()
            
            # Update visualization periodically
            if step % display_interval == 0:
                self.visualizer.update_plots()
                plt.pause(0.01)
            
            # Print status
            current_time = self.environment.time.strftime('%Y-%m-%d %H:%M')
            current_energy = self.environment.energy_consumption
            if step % steps_per_hour == 0:
                print(f"Simulation time: {current_time}, Energy usage: {current_energy:.2f} kW")
        
        # Calculate final results
        self.calculate_results()
        
        # Display final results
        self.display_results()
    
    def calculate_results(self):
        """Calculate and store final simulation results"""
        # Total energy consumption (kWh)
        self.results['energy_total'] = sum(self.environment.energy_history)
        
        # Average comfort level
        self.results['comfort_avg'] = sum(self.environment.comfort_history) / len(self.environment.comfort_history) if self.environment.comfort_history else 0
        
        # Calculate occupancy hours for each room
        occupancy_hours = 0
        for i in range(self.environment.rooms):
            room_occupancy = self.environment.history['occupancy'][i]
            occupancy_time = sum(room_occupancy) * (self.simulation_duration / len(room_occupancy))
            occupancy_hours += occupancy_time
        
        self.results['occupancy_hours'] = occupancy_hours
    
    def display_results(self):
        """Display the final simulation results"""
        print("\n===== SIMULATION RESULTS =====")
        print(f"Simulation Duration: {self.simulation_duration} hours")
        print(f"Total Energy Consumption: {self.results['energy_total']:.2f} kWh")
        print(f"Average Comfort Level: {self.results['comfort_avg']:.2f}%")
        print(f"Total Occupancy Time: {self.results['occupancy_hours']:.2f} person-hours")
        
        # Create a figure for final results
        fig, ax = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Smart Home Simulation Results', fontsize=16)
        
        # Plot temperature over time
        ax[0, 0].set_title('Temperature Over Time')
        ax[0, 0].set_xlabel('Time')
        ax[0, 0].set_ylabel('Temperature (°C)')
        times = [t.strftime('%H:%M') for t in self.environment.history['time']]
        
        for i in range(self.environment.rooms):
            ax[0, 0].plot(times, self.environment.history['temperature'][i], 
                        label=f"{self.visualizer.room_names[i]}", 
                        color=self.visualizer.room_colors[i])
        
        # Also plot outdoor temperature
        outdoor_temp = [self.environment.outdoor_temperature for _ in range(len(times))]
        ax[0, 0].plot(times, outdoor_temp, label='Outdoor', color='black', linestyle='--')
        
        ax[0, 0].set_xticks(times[::max(1, len(times)//10)])
        ax[0, 0].set_xticklabels(times[::max(1, len(times)//10)], rotation=45)
        ax[0, 0].legend()
        ax[0, 0].grid(True)
        
        # Plot energy consumption
        ax[0, 1].set_title('Cumulative Energy Consumption')
        ax[0, 1].set_xlabel('Time')
        ax[0, 1].set_ylabel('Energy (kWh)')
        
        cumulative_energy = np.cumsum(self.environment.energy_history)
        ax[0, 1].plot(times, cumulative_energy, color='orange')
        ax[0, 1].set_xticks(times[::max(1, len(times)//10)])
        ax[0, 1].set_xticklabels(times[::max(1, len(times)//10)], rotation=45)
        ax[0, 1].grid(True)
        
        # Plot comfort level
        ax[1, 0].set_title('Comfort Level Over Time')
        ax[1, 0].set_xlabel('Time')
        ax[1, 0].set_ylabel('Comfort (%)')
        
        ax[1, 0].plot(times, self.environment.comfort_history, color='green')
        ax[1, 0].set_xticks(times[::max(1, len(times)//10)])
        ax[1, 0].set_xticklabels(times[::max(1, len(times)//10)], rotation=45)
        ax[1, 0].set_ylim(0, 100)
        ax[1, 0].grid(True)
        
        # Plot occupancy
        ax[1, 1].set_title('Room Occupancy')
        ax[1, 1].set_xlabel('Time')
        ax[1, 1].set_ylabel('Occupied (1=Yes)')
        
        for i in range(self.environment.rooms):
            # Offset the occupancy lines slightly so they don't overlap
            offset = i * 0.05
            occupancy_data = [occ + offset for occ in self.environment.history['occupancy'][i]]
            ax[1, 1].step(times, occupancy_data, 
                        label=f"{self.visualizer.room_names[i]}", 
                        color=self.visualizer.room_colors[i], 
                        where='post')
        
        ax[1, 1].set_xticks(times[::max(1, len(times)//10)])
        ax[1, 1].set_xticklabels(times[::max(1, len(times)//10)], rotation=45)
        ax[1, 1].set_yticks([0, 1])
        ax[1, 1].set_yticklabels(['Unoccupied', 'Occupied'])
        ax[1, 1].legend()
        ax[1, 1].grid(True)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig('simulation_results.png', dpi=300)
        plt.show()
    
    def save_results_to_csv(self, filename='simulation_data.csv'):
        """Save the simulation data to a CSV file"""
        # Create a DataFrame with all the data
        data = {
            'time': self.environment.history['time'],
            'outdoor_temperature': [self.environment.outdoor_temperature for _ in range(len(self.environment.history['time']))]
        }
        
        # Add room-specific data
        for i in range(self.environment.rooms):
            room_name = self.visualizer.room_names[i]
            data[f'{room_name}_temperature'] = self.environment.history['temperature'][i]
            data[f'{room_name}_luminosity'] = self.environment.history['luminosity'][i]
            data[f'{room_name}_air_quality'] = self.environment.history['air_quality'][i]
            data[f'{room_name}_occupancy'] = self.environment.history['occupancy'][i]
        
        # Add energy and comfort data
        data['energy_consumption'] = self.environment.energy_history
        data['comfort'] = self.environment.comfort_history
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Simulation data saved to {filename}")


def main():
    """Main function to run the simulation"""
    print("Starting Smart Home Environmental Control Simulation")
    
    # Create and run simulation
    simulation = SmartHomeSimulation(rooms=4, simulation_duration=24)
    simulation.run(steps_per_hour=6, display_interval=2)
    
    # Save results
    simulation.save_results_to_csv()
    
    print("Simulation complete")


if __name__ == "__main__":
    main()