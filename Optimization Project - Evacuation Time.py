import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define the building parameters
num_floors = 10
num_people = 200
stairs_percentage = 0.3  # Percentage of people using stairs

# Randomly distribute people in the building
people_per_floor = np.random.multinomial(num_people, [1/num_floors] * num_floors)

# Calculate the total number of people on each floor
people_cumulative = np.cumsum(people_per_floor)

# Define the time it takes for one person to evacuate a floor (assumed constant)
evacuation_time = 30  # in seconds

# Define additional factors for realistic evacuation time
random_factor = np.random.uniform(0.8, 1.2, size=num_floors)  # Occupant behavior factor
congestion_delay = np.sqrt(people_per_floor) * 0.5  # Congestion delay based on number of people
obstacle_factor = np.random.uniform(0.9, 1.1, size=num_floors)  # Obstacle factor per floor
stair_capacity = 20  # Stair capacity limit (number of people)
elevator_capacity = 10  # Elevator capacity limit (number of people)
pre_evacuation_time = 5  # Pre-evacuation time in seconds
emergency_system_factor = 1.1  # Emergency systems factor
coordination_factor = 2  # Coordination factor
disability_delay = np.random.uniform(0, 2, size=num_floors)  # Additional delay for people with disabilities

# Define the objective function to minimize (total evacuation time)
def objective(x):
    stairs_time = x[0]
    lift_time = x[1]
    
    # Calculate the total evacuation time using the given times for stairs and lift
    total_time = np.maximum(people_cumulative * stairs_time * random_factor * obstacle_factor,
                             people_cumulative * lift_time + (people_per_floor - people_cumulative) * stairs_time * random_factor * obstacle_factor)
    
    # Add additional factors to the total evacuation time
    total_time += congestion_delay
    total_time += pre_evacuation_time
    total_time *= emergency_system_factor
    total_time += coordination_factor
    total_time += disability_delay
    
    return np.sum(total_time)

# Define the constraint that the sum of stairs_percentage and lift_percentage should be 1
def constraint(x):
    return 1 - np.sum(x)

# Initial guess for the optimization variables
x0 = [0.5, 0.5]  # Initial guess for stairs_time and lift_time

# Define the bounds for the optimization variables
bounds = [(0, None), (0, None)]  # No upper bound for the times, but they cannot be negative

# Perform the optimization
result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints={'type': 'eq', 'fun': constraint})

# Extract the optimized times
optimized_times = result.x

# Calculate the final evacuation time using the optimized times
final_evacuation_time = objective(optimized_times)

# Print the optimized times and the final evacuation time
print("Optimized Times (Stairs, Lift):", optimized_times, "seconds")
print("Final Evacuation Time:", final_evacuation_time, "seconds")

# Print the distribution of people on each floor
print("Distribution of People on Each Floor:", people_per_floor)

# Print the percentage of people using stairs vs. lift
print("Percentage of People Using Stairs: {:.2%}".format(stairs_percentage))
print("Percentage of People Using Lift: {:.2%}".format(1 - stairs_percentage))

# Calculate the evacuation time per floor for stairs and lift
stairs_time_per_floor = people_per_floor * optimized_times[0] * random_factor * obstacle_factor
lift_time_per_floor = people_cumulative * optimized_times[1] + (people_per_floor - people_cumulative) * optimized_times[0] * random_factor * obstacle_factor

# Print the evacuation time per floor for stairs and lift
print("Evacuation Time per Floor (Stairs):", stairs_time_per_floor, "seconds")
print("Evacuation Time per Floor (Lift):", lift_time_per_floor, "seconds")

# Calculate the cumulative evacuation time
cumulative_evacuation_time = np.cumsum(stairs_time_per_floor)

# Print the cumulative evacuation time
print("Cumulative Evacuation Time:", cumulative_evacuation_time, "seconds")

# Calculate the evacuation rate per floor
evacuation_rate = people_per_floor / stairs_time_per_floor

# Print the evacuation rate per floor
print("Evacuation Rate per Floor:", evacuation_rate, "people/second")

# Calculate the evacuation time distribution
evacuation_times = np.maximum(stairs_time_per_floor, lift_time_per_floor)

# Print the evacuation time distribution
print("Evacuation Time Distribution:", evacuation_times, "seconds")

# Plotting the evacuation curve
time_per_floor = np.maximum(people_per_floor * optimized_times[0] * random_factor * obstacle_factor,
                            people_cumulative * optimized_times[1] + (people_per_floor - people_cumulative) * optimized_times[0] * random_factor * obstacle_factor)

plt.plot(range(1, num_floors + 1), time_per_floor, marker='o')
plt.xlabel("Floor")
plt.ylabel("Evacuation Time (seconds)")
plt.title("Evacuation Time per Floor")
plt.grid(True)
plt.show()

plt.bar(range(1, num_floors + 1), people_per_floor)
plt.xlabel("Floor")
plt.ylabel("Number of People")
plt.title("Distribution of People on Each Floor")
plt.grid(True)
plt.show()

labels = ['Stairs', 'Lift']
sizes = [stairs_percentage, 1 - stairs_percentage]

plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True)
plt.title("Percentage of People Using Stairs vs. Lift")
plt.axis('equal')
plt.show()

plt.plot(range(1, num_floors + 1), stairs_time_per_floor, label='Stairs', marker='o')
plt.plot(range(1, num_floors + 1), lift_time_per_floor, label='Lift', marker='o')
plt.xlabel("Floor")
plt.ylabel("Evacuation Time (seconds)")
plt.title("Evacuation Time Comparison (Stairs vs. Lift)")
plt.legend()
plt.grid(True)
plt.show()

plt.plot(range(1, num_floors + 1), cumulative_evacuation_time, marker='o')
plt.xlabel("Floor")
plt.ylabel("Cumulative Evacuation Time (seconds)")
plt.title("Cumulative Evacuation Time")
plt.grid(True)
plt.show()

evacuation_rate = people_per_floor / time_per_floor

plt.plot(range(1, num_floors + 1), evacuation_rate, marker='o')
plt.xlabel("Floor")
plt.ylabel("Evacuation Rate (people/second)")
plt.title("People Evacuated per Unit Time")
plt.grid(True)
plt.show()

evacuation_times = np.maximum(people_per_floor * optimized_times[0] * random_factor * obstacle_factor,
                              people_cumulative * optimized_times[1] + (people_per_floor - people_cumulative) * optimized_times[0] * random_factor * obstacle_factor)

plt.hist(evacuation_times, bins='auto')
plt.xlabel("Evacuation Time (seconds)")
plt.ylabel("Frequency")
plt.title("Evacuation Time Distribution")
plt.grid(True)
plt.show()

evacuation_rate = people_per_floor / stairs_time_per_floor

plt.plot(range(1, num_floors + 1), evacuation_rate, marker='o')
plt.xlabel("Floor")
plt.ylabel("Evacuation Rate (people/second)")
plt.title("Evacuation Rate per Floor")
plt.grid(True)
plt.show()


plt.bar(['Minimum Evacuation Time'], [final_evacuation_time])
plt.xlabel('Result')
plt.ylabel('Evacuation Time (seconds)')
plt.title('Minimum Evacuation Time')
plt.grid(True)
plt.show()































