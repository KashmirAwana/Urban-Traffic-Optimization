import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import networkx as nx
import random
import time
from collections import deque, defaultdict
from datetime import datetime, timedelta
from matplotlib.widgets import Button, Slider

class TrafficLight:
    def _init_(self, location_id, coordinates):
        self.location_id = location_id
        self.coordinates = coordinates  # (x, y) for visualization
        self.neighbors = []  # List of neighboring traffic light IDs
        self.current_direction = 0  # 0-3 representing the four directions
        self.green_duration = 30  # seconds
        self.yellow_duration = 5  # seconds
        self.current_state = "green"  # green, yellow, red 
        self.state_time_left = self.green_duration
        self.traffic_counts = {i: 0 for i in range(4)}  # Cars waiting in each direction
        self.historical_data = defaultdict(lambda: defaultdict(list))  # Time of day -> direction -> [counts]
        self.emergency_mode = False
        self.accident_location = None
        self.display_message = None
        self.sensors = [0] * 4  # Sensor readings for incoming traffic from each direction
        
        # For reinforcement learning
        self.q_table = defaultdict(lambda: np.zeros(4))  # State -> action values
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.2
        
        # Direction names for clarity
        self.direction_names = ["North", "East", "South", "West"]
        
    def add_neighbor(self, neighbor_id, direction):
        """Add a neighboring traffic light"""
        self.neighbors.append((neighbor_id, direction))
    
    def get_state(self):
        """Get the current state for RL (simplified)"""
        # State is a tuple of traffic counts in each direction
        return tuple(self.traffic_counts.values())
    
    def choose_action(self):
        """Choose which direction to give green light to next"""
        state = self.get_state()
        
        # Exploration-exploitation tradeoff
        if random.random() < self.exploration_rate:
            return random.randint(0, 3)  # Random direction
        else:
            return np.argmax(self.q_table[state])
    
    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value for a state-action pair"""
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        
        # Q-learning formula
        new_value = old_value + self.learning_rate * (reward + self.discount_factor * next_max - old_value)
        self.q_table[state][action] = new_value
    
    def update_state(self, dt, current_time):
        """Update the traffic light state based on time passed"""
        hour = current_time.hour + current_time.minute / 60
        
        # Record current traffic for historical data
        time_bucket = int(hour)
        for direction, count in self.traffic_counts.items():
            self.historical_data[time_bucket][direction].append(count)
        
        self.state_time_left -= dt
        
        if self.state_time_left <= 0:
            if self.current_state == "green":
                self.current_state = "yellow"
                self.state_time_left = self.yellow_duration
            elif self.current_state == "yellow":
                self.current_state = "green"
                
                # Choose next direction based on RL
                old_state = self.get_state()
                old_direction = self.current_direction
                
                if self.emergency_mode and self.accident_location is not None:
                    # Prioritize directions away from accident
                    accident_dir = self.accident_location
                    opposite_dir = (accident_dir + 2) % 4  # Opposite in 4-way intersection
                    self.current_direction = opposite_dir
                else:
                    self.current_direction = self.choose_action()
                
                # Adaptive green light duration based on traffic volume
                base_duration = 30
                traffic_factor = min(3.0, max(0.5, self.traffic_counts[self.current_direction] / 10))
                self.green_duration = int(base_duration * traffic_factor)
                
                # Adjust for time of day patterns
                if time_bucket in [7, 8, 9, 16, 17, 18]:  # Rush hours
                    self.green_duration = int(self.green_duration * 1.2)  # 20% longer during rush hours
                
                self.state_time_left = self.green_duration
                
                # Calculate reward (negative of total waiting cars)
                reward = -sum(self.traffic_counts.values())
                new_state = self.get_state()
                
                # Update Q-values
                self.update_q_value(old_state, old_direction, reward, new_state)
    
    def receive_traffic_data(self, direction, count):
        """Receive traffic data from a neighboring traffic light"""
        # Update expected incoming traffic
        self.sensors[direction] = count
    
    def declare_emergency(self, accident_direction):
        """Traffic police declares emergency due to accident"""
        self.emergency_mode = True
        self.accident_location = accident_direction
        self.display_message = f"Accident on {self.direction_names[accident_direction]} side. Avoid this route."
        
        # Notify neighbors about the accident
        return self.location_id, accident_direction
    
    def end_emergency(self):
        """End emergency mode"""
        self.emergency_mode = False
        self.accident_location = None
        self.display_message = None
    
    def process_vehicles(self, dt):
        """Process vehicles passing through the intersection"""
        vehicles_passed = 0
        
        if self.current_state == "green":
            # Cars can pass through in the current green direction
            flow_rate = 2  # cars per second
            max_cars = int(flow_rate * dt)
            cars_passed = min(max_cars, self.traffic_counts[self.current_direction])
            self.traffic_counts[self.current_direction] -= cars_passed
            vehicles_passed = cars_passed
        
        # Completely remove random vehicle generation initially
        # Only add vehicles during emergencies or traffic surges
        return vehicles_passed

class TrafficSystem:
    def _init_(self, grid_size=3):
        self.grid_size = grid_size
        self.traffic_lights = {}
        self.emergency_notifications = []
        self.current_time = datetime(2025, 2, 28, 8, 0)  # Starting at 8:00 AM
        self.historical_data = pd.DataFrame()  # For tracking system performance
        self.simulation_speed = 1.0  # Default speed
        self.paused = False  # Simulation state
        self.events_log = []  # Log of important events
        self.event_time_display = 5  # How long events stay displayed (in seconds)
        
        # Track arrows for proper cleanup
        self.emergency_arrows = []
        
        # Store pre-emergency traffic state
        self.pre_emergency_traffic = {}
        
        # Create hexagonal grid network
        self.create_hexagonal_grid()
        
        # Create some historical data
        self.generate_historical_data()
        
        # Add initial traffic to make visualization more interesting
        self.initialize_traffic()
        
        # Initialize a traffic surge on startup to demonstrate
        self.surge_active = False
        self.surge_location = None
        self.surge_direction = None
        self.surge_end_time = None
        self.next_surge_time = self.current_time + timedelta(seconds=15)  # Start surge after 15 seconds
        self.surge_duration = timedelta(minutes=0)
    
    def create_hexagonal_grid(self):
        """Create a hexagonal grid of traffic lights"""
        # For a smaller grid with 4-way intersections instead of 6-way
        directions = [
            (0, 1),      # 0: North
            (1, 0),      # 1: East
            (0, -1),     # 2: South
            (-1, 0)      # 3: West
        ]
        
        # Create traffic lights in a hexagonal pattern but with 4-way intersections
        count = 0
        for q in range(-self.grid_size, self.grid_size + 1):
            for r in range(-self.grid_size, self.grid_size + 1):
                # Skip some positions to create a hexagonal-like pattern
                if abs(q + r) > self.grid_size:
                    continue
                
                # Convert to cartesian coordinates
                x = q * 1.5
                y = r * 1.5
                
                # Create traffic light
                self.traffic_lights[count] = TrafficLight(count, (x, y))
                count += 1
        
        # Connect neighboring traffic lights
        for light_id, light in self.traffic_lights.items():
            x, y = light.coordinates
            
            for direction, (dx, dy) in enumerate(directions):
                # Find neighbor at this direction
                nx, ny = x + dx * 1.5, y + dy * 1.5
                
                # Find closest traffic light to this point
                closest_id = None
                min_dist = float('inf')
                
                for other_id, other_light in self.traffic_lights.items():
                    if other_id != light_id:
                        ox, oy = other_light.coordinates
                        dist = np.sqrt((nx - ox)*2 + (ny - oy)*2)
                        if dist < 2.0:  # Threshold
                            min_dist = dist
                            closest_id = other_id
                
                # If neighbor is close enough, add it
                if closest_id is not None:
                    light.add_neighbor(closest_id, direction)
    
    def generate_historical_data(self):
        """Generate some historical traffic data"""
        for light in self.traffic_lights.values():
            for hour in range(24):
                for direction in range(4):
                    # Create typical traffic patterns
                    if 7 <= hour <= 9:  # Morning rush
                        avg_traffic = 15 + random.randint(0, 10)
                    elif 16 <= hour <= 18:  # Evening rush
                        avg_traffic = 18 + random.randint(0, 15)
                    elif 11 <= hour <= 13:  # Lunch
                        avg_traffic = 10 + random.randint(0, 8)
                    elif 22 <= hour <= 23 or 0 <= hour <= 5:  # Night
                        avg_traffic = 2 + random.randint(0, 3)
                    else:  # Normal daytime
                        avg_traffic = 7 + random.randint(0, 5)
                    
                    # Add random variation
                    for _ in range(30):  # 30 samples per hour
                        value = max(0, int(np.random.normal(avg_traffic, avg_traffic/3)))
                        light.historical_data[hour][direction].append(value)
    
    def simulate_step(self, dt=1.0):
        if self.paused:
            return 0, 0
        
        # Update time
        self.current_time += timedelta(seconds=dt * self.simulation_speed)
        
        # Only trigger traffic surge and vehicle generation during emergencies
        if self.emergency_notifications:
            # Check for traffic surge events
            if self.current_time >= self.next_surge_time and not self.surge_active:
                self.generate_traffic_surge()
            
            # Update active surge
            if self.surge_active:
                if self.current_time >= self.surge_end_time:
                    self.end_traffic_surge()
                else:
                    self.apply_traffic_surge()
        
        # Process traffic at each light
        total_waiting = 0
        total_passed = 0
        
        # First, collect traffic data from all intersections
        traffic_data = {}
        for light_id, light in self.traffic_lights.items():
            traffic_data[light_id] = {dir: light.traffic_counts[dir] for dir in range(4)}
        
        # Share data with neighbors
        for light_id, light in self.traffic_lights.items():
            for neighbor_id, direction in light.neighbors:
                if neighbor_id in self.traffic_lights:
                    # The direction from neighbor to this light
                    incoming_dir = (direction + 2) % 4  # Opposite direction in 4-way
                    # Share traffic count in the outgoing direction
                    self.traffic_lights[neighbor_id].receive_traffic_data(
                        incoming_dir, light.traffic_counts[direction]
                    )
        
        # Update each traffic light
        for light_id, light in self.traffic_lights.items():
            light.update_state(dt * self.simulation_speed, self.current_time)
            vehicles_passed = light.process_vehicles(dt * self.simulation_speed)
            
            total_waiting += sum(light.traffic_counts.values())
            total_passed += vehicles_passed
        
        # Record performance metrics
        new_row = {
            'time': self.current_time,
            'total_waiting_vehicles': total_waiting,
            'vehicles_passed': total_passed,
            'emergency_count': len(self.emergency_notifications),
            'average_wait_per_intersection': total_waiting / max(1, len(self.traffic_lights))
        }
        self.historical_data = pd.concat([self.historical_data, pd.DataFrame([new_row])], ignore_index=True)
        
        # Clean up old events
        self.clean_old_events()
        
        return total_waiting, total_passed
    
    def initialize_traffic(self):
        """Initialize with some traffic to make visualization interesting from the start"""
        for light_id, light in self.traffic_lights.items():
            # Add random initial traffic (more controlled)
            for direction in range(4):
                # Introduce some variability but keep it moderate
                if random.random() < 0.2:  # 20% chance of high traffic
                    light.traffic_counts[direction] = random.randint(30, 40)
                elif random.random() < 0.4:  # 40% chance of moderate traffic
                    light.traffic_counts[direction] = random.randint(15, 29)
                else:  # 40% chance of light traffic
                    light.traffic_counts[direction] = random.randint(5, 14)
        
        # Save initial traffic state as pre-emergency state
        for l_id, light in self.traffic_lights.items():
            self.pre_emergency_traffic[l_id] = {dir: count for dir, count in light.traffic_counts.items()}
    
    def generate_traffic_surge(self):
        """Generate a traffic surge event near an emergency"""
        if not self.emergency_notifications:
            return

        self.surge_active = True
        self.surge_location = random.choice([n[0] for n in self.emergency_notifications])
        self.surge_direction = random.randint(0, 3)
        
        self.surge_duration = timedelta(minutes=random.randint(1, 3))
        self.surge_end_time = self.current_time + self.surge_duration
        
        if self.surge_location in self.traffic_lights:
            surge_light = self.traffic_lights[self.surge_location]
            
            # Spread traffic increase across multiple directions
            for direction in range(4):
                current_count = surge_light.traffic_counts[direction]
                # Moderate traffic increase
                increase = random.randint(10, 20)
                surge_light.traffic_counts[direction] = min(current_count + increase, 50)
                
            # Log the event
            event_msg = f"Traffic surge detected near intersection {self.surge_location} on the {surge_light.direction_names[self.surge_direction]} side."
            self.add_event(event_msg)
    
    def apply_traffic_surge(self):
        """Apply a more noticeable traffic surge"""
        if not self.surge_active:
            return
            
        # Add extra traffic to the surge location and neighboring intersections
        if self.surge_location in self.traffic_lights:
            # Increase surge factor for more noticeable effect
            surge_factor = 8.0  # Increased from 5.0
            light = self.traffic_lights[self.surge_location]
            
            # Add more vehicles in surge direction
            extra_cars = int(random.uniform(8, 15) * self.simulation_speed)  # Increased from 5-10
            light.traffic_counts[self.surge_direction] += extra_cars
            
            # Propagate to neighbors
            for neighbor_id, direction in light.neighbors:
                if neighbor_id in self.traffic_lights:
                    if direction == self.surge_direction:
                        neighbor_dir = (direction + 2) % 4  # Opposite direction
                        self.traffic_lights[neighbor_id].traffic_counts[neighbor_dir] += int(extra_cars * 0.7)
    
    def end_traffic_surge(self):
        """End the current traffic surge"""
        if self.surge_active:
            event_msg = f"Traffic surge at intersection {self.surge_location} has subsided."
            self.add_event(event_msg)
            self.surge_active = False
            
            # Set the next surge time
            self.next_surge_time = self.current_time + timedelta(minutes=random.randint(1, 5))
    
    def declare_emergency(self, light_id, accident_direction):
        """Declare emergency at a specific traffic light"""
        if light_id in self.traffic_lights:
            # Save traffic state before first emergency if not already saved
            if not self.pre_emergency_traffic:
                for l_id, light in self.traffic_lights.items():
                    self.pre_emergency_traffic[l_id] = {dir: count for dir, count in light.traffic_counts.items()}
            
            notification = self.traffic_lights[light_id].declare_emergency(accident_direction)
            self.emergency_notifications.append(notification)
            
            # Log the event
            direction_name = self.traffic_lights[light_id].direction_names[accident_direction]
            event_msg = f"EMERGENCY: Accident at intersection {light_id} on the {direction_name} side. Traffic being rerouted."
            self.add_event(event_msg)
            
            # Trigger a traffic surge
            self.generate_traffic_surge()
            
            # Notify neighbors
            for neighbor_id, _ in self.traffic_lights[light_id].neighbors:
                if neighbor_id in self.traffic_lights:
                    opposite_dir = (accident_direction + 2) % 4
                    neighbor_direction = self.traffic_lights[neighbor_id].direction_names[opposite_dir]
                    self.traffic_lights[neighbor_id].display_message = f"Accident near intersection {light_id}. Avoid {neighbor_direction} direction."
    
    def end_emergency(self, light_id):
        """End emergency at a specific traffic light"""
        if light_id in self.traffic_lights:
            self.traffic_lights[light_id].end_emergency()
            
            # Log the event
            event_msg = f"Emergency situation at intersection {light_id} has been resolved."
            self.add_event(event_msg)
            
            # Remove from notifications
            self.emergency_notifications = [n for n in self.emergency_notifications if n[0] != light_id]
            
            # Clear messages from neighbors
            for neighbor_id, _ in self.traffic_lights[light_id].neighbors:
                if neighbor_id in self.traffic_lights:
                    self.traffic_lights[neighbor_id].display_message = None
    
    def clear_emergency_traffic(self, light_id):
        """Rapidly clear traffic caused by an emergency situation"""
        if light_id not in self.traffic_lights:
            return
        
        # If we have pre-emergency traffic data, restore it
        if self.pre_emergency_traffic:
            for l_id, directions in self.pre_emergency_traffic.items():
                if l_id in self.traffic_lights:
                    for direction, count in directions.items():
                        self.traffic_lights[l_id].traffic_counts[direction] = count
        else:
            # Calculate normal baseline traffic based on time of day (fallback)
            hour = self.current_time.hour
            if 7 <= hour <= 9 or 16 <= hour <= 18:  # Rush hours
                baseline = random.randint(5, 10)
            else:  # Non-rush hours
                baseline = random.randint(2, 5)
                
            # Reset all intersections to baseline
            for light in self.traffic_lights.values():
                for direction in range(4):
                    light.traffic_counts[direction] = baseline
        
        # Reset surge-related states
        self.surge_active = False
        self.surge_location = None
        self.surge_direction = None
        
        # Add an event log message
        event_msg = f"Emergency traffic cleared. Traffic restored to normal state."
        self.add_event(event_msg)
        
        # Clear emergency notifications
        self.emergency_notifications.clear()
    
    def add_event(self, message):
        """Add an event to the log with timestamp"""
        self.events_log.append((self.current_time, message))
        print(f"{self.current_time}: {message}")  # Also print to console
    
    def clean_old_events(self):
        """Remove events that are too old from display"""
        if not self.events_log:
            return
            
        current_time = self.current_time
        self.events_log = [(t, msg) for t, msg in self.events_log 
                          if (current_time - t).total_seconds() < self.event_time_display * self.simulation_speed]
    
    def toggle_pause(self, event=None):
        """Toggle between pause and play"""
        self.paused = not self.paused
        return self.paused
    
    def set_simulation_speed(self, speed):
        """Set the simulation speed"""
        self.simulation_speed = speed
    
    def visualize(self, fig, ax1, ax2, ax3):
        """Visualize the current state of the traffic system"""
        # Clear previous plots
        ax1.clear()
        ax2.clear()
        ax3.clear()
        
        # Clear all previous arrows - FIXED REMOVAL METHOD
        for arrow in self.emergency_arrows:
            if arrow in fig.patches:
                fig.patches.remove(arrow)
        self.emergency_arrows = []
        
        # Set titles with better spacing
        ax1.set_title(f"Traffic Simulation - {self.current_time.strftime('%H:%M:%S')}", pad=10, fontsize=12)
        ax2.set_title("Traffic Metrics", pad=10, fontsize=12)
        ax3.set_title("Event Log", pad=10, fontsize=12)
        
        # Draw roads and intersections
        self.draw_traffic_grid(ax1)
        
        # Add traffic surge indicator outside the main box if active
        if self.surge_active and self.surge_location in self.traffic_lights:
            # Add a prominent notification at the top of the visualization
            fig.text(0.5, 0.97, "⚠ TRAFFIC SURGE ACTIVE", fontsize=14, 
                    ha='center', va='top', color='purple', weight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
            
            # Add detailed location information
            if self.surge_location in self.traffic_lights:
                direction_name = self.traffic_lights[self.surge_location].direction_names[self.surge_direction]
                location_text = f"Location: Intersection {self.surge_location}, {direction_name} side"
                fig.text(0.5, 0.935, location_text, fontsize=12, 
                        ha='center', va='top', color='purple',
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
        
        # Display emergency notifications outside the main box
        if self.emergency_notifications:
            # Position emergency notifications on the right side
            for idx, (light_id, accident_direction) in enumerate(self.emergency_notifications[:3]):  # Limit to 3 visible notifications
                if light_id in self.traffic_lights:
                    light = self.traffic_lights[light_id]
                    direction_name = light.direction_names[accident_direction]
                    
                    # Calculate position for the notification
                    y_pos = 0.85 - (idx * 0.12)  # Vertical spacing between notifications
                    
                    # Add accident notification outside the main visualization
                    notification = fig.text(0.05, y_pos, f"⚠ ACCIDENT: Intersection {light_id}\n{direction_name} road", 
                            fontsize=10, ha='left', va='center', color='red', weight='bold',
                            bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))
                    
                    # Draw an arrow connecting the notification to the accident location
                    map_coords = light.coordinates
                    ax_pos = ax1.get_position()
                    
                    # Convert map coordinates to figure coordinates
                    map_x, map_y = map_coords
                    max_coord = (self.grid_size + 1) * 2
                    
                    # Transform from map coordinates to axes coordinates
                    ax_x = (map_x - (-max_coord)) / (2 * max_coord)
                    ax_y = (map_y - (-max_coord)) / (2 * max_coord)
                    
                    # Transform from axes to figure coordinates
                    fig_x = ax_pos.x0 + ax_x * ax_pos.width
                    fig_y = ax_pos.y0 + ax_y * ax_pos.height
                    
                    # Draw arrow connecting notification to accident location
                    arrow = patches.FancyArrowPatch(
                        (0.14, y_pos),  # Start at the end of the notification box
                        (fig_x - 0.02, fig_y),  # End near the accident location
                        connectionstyle="arc3,rad=0.2",
                        arrowstyle="simple,head_width=5,head_length=10",
                        color='red',
                        alpha=0.7,
                        transform=fig.transFigure,
                        figure=fig
                    )
                    fig.patches.append(arrow)
                    self.emergency_arrows.append(arrow)  # Track the arrow for later removal
        
        # Plot metrics if we have data
        if len(self.historical_data) > 1:
            self.plot_metrics(ax2)
        
        # Display event log
        self.display_event_log(ax3)
        
        # Add legend to the traffic grid
        self.add_legend(fig)
        
        # Adjust layout to prevent overlapping
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        return fig
    
    def draw_traffic_grid(self, ax):
        # Draw roads first (as lines)
        for light_id, light in self.traffic_lights.items():
            x, y = light.coordinates
            
            # Draw roads to neighbors first (background)
            for neighbor_id, direction in light.neighbors:
                if neighbor_id in self.traffic_lights:
                    nx, ny = self.traffic_lights[neighbor_id].coordinates
                    # Draw road with width
                    self.draw_road(ax, x, y, nx, ny)
        
        # Now draw traffic lights without boxes behind them
        for light_id, light in self.traffic_lights.items():
            x, y = light.coordinates
            
            # Draw traffic light as a circle with clearer edges
            state_color = light.current_state  # green, yellow, or red
            traffic_light = patches.Circle((x, y), 0.2, linewidth=2,
                                    edgecolor='black', facecolor=state_color, zorder=10)
            ax.add_patch(traffic_light)
            
            # Draw traffic in each direction
            for direction in range(4):
                # Direction vectors for 4-way intersection
                if direction == 0:  # North
                    dx, dy = 0, 0.5
                elif direction == 1:  # East
                    dx, dy = 0.5, 0
                elif direction == 2:  # South
                    dx, dy = 0, -0.5
                else:  # West
                    dx, dy = -0.5, 0
                
                # Position on the road
                tx = x + dx
                ty = y + dy
                
                # Show traffic as squares, size based on count, color based on amount
                traffic_count = light.traffic_counts[direction]
                size = min(0.3, max(0.05, 0.05 + traffic_count * 0.01))
                
                # Color scale from green to red based on traffic
                if traffic_count < 5:
                    bg_color = 'green'
                    text_color = 'white'
                elif traffic_count < 10:
                    bg_color = 'yellow'
                    text_color = 'black'
                elif traffic_count < 20:
                    bg_color = 'orange'
                    text_color = 'white'
                else:
                    bg_color = 'red'
                    text_color = 'white'
                
                # Show count in a single colored box with no internal lines
                ax.text(tx, ty, str(traffic_count), fontsize=10, ha='center', va='center', 
                    color=text_color, weight='bold', 
                    bbox=dict(facecolor=bg_color, edgecolor='black', linewidth=1.5, 
                             boxstyle='round,pad=0.3', alpha=0.9),
                    zorder=6)
                
                # Highlight active green direction with clearer arrow
                if direction == light.current_direction and light.current_state == "green":
                    arrow_dx = dx * 0.3
                    arrow_dy = dy * 0.3
                    arrow = patches.FancyArrow(x, y, arrow_dx, arrow_dy, width=0.05,
                                        head_width=0.15, head_length=0.1,
                                        facecolor='lime', edgecolor='black', zorder=15)
                    ax.add_patch(arrow)
            
            # Show intersection ID below the traffic light
            ax.text(x, y-0.3, str(light_id), fontsize=8, ha='center', va='center', 
                    color='black', weight='bold', zorder=11)
            
            # Show emergency message if any
            if light.display_message:
                # Only show warning symbol on the map
                ax.text(x, y+0.4, "⚠", fontsize=14, ha='center', va='center', 
                        color='red', weight='bold', zorder=12)
        
        # Set axis limits
        max_coord = (self.grid_size + 1) * 2
        ax.set_xlim(-max_coord, max_coord)
        ax.set_ylim(-max_coord, max_coord)
        ax.set_aspect('equal')
        
        # Remove axis ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
    
    def draw_road(self, ax, x1, y1, x2, y2):
        """Draw a road connecting two intersections"""
        # Draw the road as a gray line with width
        road = plt.Line2D([x1, x2], [y1, y2], linewidth=3, color='gray', alpha=0.7)
        ax.add_line(road)
        
        # Draw lane markers
        self.draw_lane_markers(ax, x1, y1, x2, y2)
    
    def draw_lane_markers(self, ax, x1, y1, x2, y2):
        """Draw lane markers on a road"""
        # Calculate the direction vector
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx*2 + dy*2)
        
        # Prevent division by zero
        if length == 0:
            return
        
        dx, dy = dx/length, dy/length
        
        # Perpendicular vector - fixed the calculation
        px, py = -dy*0.03, dx*0.03
        
        # Number of markers
        num_markers = int(length / 0.2)
        
        # Draw dashed line in the middle
        for i in range(1, num_markers):
            t = i / num_markers
            if i % 2 == 0:  # Skip every other to make it dashed
                continue
            # Middle point
            mx = x1 + dx * length * t
            my = y1 + dy * length * t
            # Draw a small line segment
            marker = plt.Line2D([mx-px, mx+px], [my-py, my+py], linewidth=1, color='white')
            ax.add_line(marker)
    
    def plot_metrics(self, ax):
        """Plot traffic metrics over time with enhanced visibility and spacing"""
        data = self.historical_data.copy()
        
        # Limit data points for efficiency
        if len(data) > 500:
            data = data.iloc[len(data)-500:]
        
        # Get time values in seconds for x-axis
        start_time = data['time'].iloc[0]
        time_seconds = [(t - start_time).total_seconds() for t in data['time']]
        
        # Plot waiting vehicles with thicker line and brighter color
        ax.plot(time_seconds, data['total_waiting_vehicles'], 'r-', 
                label='Waiting Vehicles', linewidth=3, color='red')
        
        # Plot vehicles passed (cumulative sum to show growth)
        cumulative_passed = data['vehicles_passed'].cumsum()
        # Scale down to fit on same axis but ensure it's visible
        scale_factor = max(data['total_waiting_vehicles'].max() / max(1, cumulative_passed.max()), 0.2)
        ax.plot(time_seconds, cumulative_passed * scale_factor, 'g-', 
                label='Vehicles Passed (scaled)', linewidth=2)
        
        # Plot emergency count with increased visibility
        if data['emergency_count'].max() > 0:
            # Scale to make it visible
            emergency_scale = max(data['total_waiting_vehicles'].max() / max(1, data['emergency_count'].max()), 0.5)
            ax.plot(time_seconds, data['emergency_count'] * emergency_scale, 'y--', 
                    label='Emergencies (scaled)', linewidth=2)
        
        # Add markers for traffic surges
        surge_times = []
        for i, (t, msg) in enumerate(self.events_log):
            if "Traffic surge" in msg:
                surge_time = (t - start_time).total_seconds()
                if surge_time >= 0 and surge_time <= time_seconds[-1]:
                    surge_times.append(surge_time)
        
        if surge_times:
            # Get the y-values at these times by interpolating
            max_y = data['total_waiting_vehicles'].max()
            ax.scatter(surge_times, [max_y * 0.9] * len(surge_times), 
                    color='purple', s=100, marker='^', label='Traffic Surges')
        
        # Format the x-axis with time labels
        if len(time_seconds) > 0:
            # Show minutes and seconds
            def format_time(seconds):
                m, s = divmod(int(seconds), 60)
                return f"{m:02d}:{s:02d}"
            
            # Set x-ticks at regular intervals with better spacing
            interval = max(1, len(time_seconds) // 5)
            ticks = time_seconds[::interval]
            ax.set_xticks(ticks)
            ax.set_xticklabels([format_time(t) for t in ticks], rotation=45)
        
        # Ensure y-axis starts from 0 to show the full scale
        ax.set_ylim(bottom=0)
        
        # Add gridlines for better readability
        ax.grid(True, alpha=0.4, linestyle='--')
        
        # Add padding around labels
        ax.set_xlabel("Simulation Time (mm:ss)", labelpad=10)
        ax.set_ylabel("Vehicle Count", labelpad=10)
        
        # Better legend placement with padding
        ax.legend(loc='upper left', framealpha=0.7, frameon=True, fancybox=True, shadow=True, borderpad=1)
    
    def display_event_log(self, ax):
        """Display the event log with improved spacing and readability"""
        # Clear the axes and set axis off for text display
        ax.axis('off')
        
        # Display the most recent events (limited to prevent overcrowding)
        max_events = 5
        recent_events = self.events_log[-max_events:] if len(self.events_log) > 0 else []
        
        # Add title with better styling
        ax.text(0.5, 0.98, "Event Log", transform=ax.transAxes,
                fontsize=12, fontweight='bold', ha='center')
        
        event_text = ""
        if not recent_events:
            event_text += "No recent events."
        else:
            for i, (time, msg) in enumerate(recent_events):
                time_str = time.strftime('%H:%M:%S')
                # Add padding between events
                event_text += f"{time_str}: {msg}\n\n"
        
        # Add the text to the axis with better padding and background
        ax.text(0.05, 0.85, event_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='wheat', alpha=0.5))

    def add_legend(self, fig):
        """Add a legend explaining the symbols used in the visualization"""
        legend_text = """Legend:
    
    • Green/Yellow/Red circles: Traffic light state
    • Green arrow: Direction of current green light
    • Colored squares: Traffic congestion (green=low, red=high)
    • Numbers in squares: Vehicle count waiting
    • ⚠: Emergency situation
    • Purple indicator: Traffic surge location
    """
        
        # Add text box for legend with improved spacing and style
        fig.text(0.02, 0.02, legend_text, fontsize=9,
                bbox=dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.8))

# Main function to run the simulation
def main():
    """Main function to run the traffic simulation"""
    # Create the traffic system
    global traffic_system
    traffic_system = TrafficSystem(grid_size=2)
    
    # Set up the visualization with better spacing and margins
    plt.ion()  # Interactive mode
    fig = plt.figure(figsize=(15, 8))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)  # Add spacing between subplots
    
    # Create subplots for traffic grid, metrics, and event log with better proportions
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
    ax2 = plt.subplot2grid((2, 3), (0, 2))
    ax3 = plt.subplot2grid((2, 3), (1, 2))
    
    # Add buttons and sliders for interactive control with improved spacing
    play_pause_ax = fig.add_axes([0.78, 0.05, 0.12, 0.05])  # Adjusted positioning
    play_pause_button = Button(play_pause_ax, 'Play/Pause')
    play_pause_button.on_clicked(traffic_system.toggle_pause)
    
    speed_ax = fig.add_axes([0.78, 0.12, 0.12, 0.03])  # Added spacing from play button
    speed_slider = Slider(speed_ax, 'Speed', 0.1, 5.0, valinit=1.0)
    speed_slider.on_changed(lambda val: traffic_system.set_simulation_speed(val))
    
    # Add a button to trigger random emergency with better spacing
    emergency_ax = fig.add_axes([0.63, 0.05, 0.12, 0.05])  # Adjusted width and position
    emergency_button = Button(emergency_ax, 'Random Emergency')
    
    def trigger_random_emergency(event):
        """Trigger a random emergency event"""
        if traffic_system.traffic_lights:
            light_id = random.choice(list(traffic_system.traffic_lights.keys()))
            accident_direction = random.randint(0, 3)
            traffic_system.declare_emergency(light_id, accident_direction)
    
    emergency_button.on_clicked(trigger_random_emergency)
    
    # Add a button to clear emergencies with consistent spacing
    clear_emergency_ax = fig.add_axes([0.63, 0.12, 0.12, 0.05])  # Consistent with emergency button
    clear_emergency_button = Button(clear_emergency_ax, 'Clear Emergencies')

    def clear_all_emergencies(event):
        """Clear all emergency situations and associated traffic"""
        for notification in list(traffic_system.emergency_notifications):
            light_id = notification[0]
            
            # First end the emergency
            traffic_system.end_emergency(light_id)
            
            # Then clear the associated traffic
            traffic_system.clear_emergency_traffic(light_id)

    clear_emergency_button.on_clicked(clear_all_emergencies)
    
    try:
        last_update = time.time()
        # Main simulation loop
        while plt.fignum_exists(fig.number):
            current_time = time.time()
            dt = current_time - last_update
            last_update = current_time
            
            # Limit dt to prevent large jumps if the app is unresponsive
            dt = min(dt, 0.25)
            
            # Simulate one step
            traffic_system.simulate_step(dt)
            
            # Update the visualization
            traffic_system.visualize(fig, ax1, ax2, ax3)
            
            # Pause to update display
            plt.pause(0.01)
            
    except KeyboardInterrupt:
        print("Simulation stopped by user")
    
    finally:
        plt.ioff()
        plt.close()

if _name_ == "_main_":
    main()