from z3 import *
import numpy as np

def calculate_metrics(schedule, preferences, n_volunteers, n_rooms, n_hours):
    """
    Calculate the metrics for a given schedule.
    
    Args:
        schedule: 3D binary array representing assignments
        preferences: Matrix with preference scores
        n_volunteers, n_rooms, n_hours: Problem dimensions
        
    Returns:
        pref_value: Total preference satisfaction value
        changes: Number of room changes in the schedule
    """
    # Calculate preference satisfaction
    pref_value = sum(
        preferences[v][r][h] * schedule[v, r, h]
        for v in range(n_volunteers)
        for r in range(n_rooms)
        for h in range(n_hours)
    )
    
    # Calculate room changes
    changes = 0
    for v in range(n_volunteers):
        for h in range(n_hours - 1):
            # Get rooms assigned in consecutive hours (if any)
            room_h = None
            room_h_next = None
            
            for r in range(n_rooms):
                if schedule[v, r, h] == 1:
                    room_h = r
                if schedule[v, r, h+1] == 1:
                    room_h_next = r
            
            # If volunteer is scheduled in both hours and rooms differ, count a change
            if room_h is not None and room_h_next is not None and room_h != room_h_next:
                changes += 1
    
    return pref_value, changes

def surveillance_scheduling(
    n_rooms,          # Number of rooms
    n_hours,          # Number of hours
    n_volunteers,     # Number of volunteers
    availability,     # Binary matrix (n_volunteers x n_hours) indicating volunteer availability
    preferences,      # Matrix (n_volunteers x n_rooms x n_hours) with preference scores
    pref_weight=0.7,  # Weight for preference objective (0-1)
    change_weight=0.3 # Weight for minimizing room changes (0-1)
):
    """
    Solves the room surveillance scheduling problem using Z3 SMT solver.
    
    Args:
        n_rooms: Number of rooms that need surveillance
        n_hours: Number of hours in the scheduling period
        n_volunteers: Number of available volunteers
        availability: Binary matrix where availability[v][h] = 1 if volunteer v is available at hour h
        preferences: Matrix where preferences[v][r][h] is the preference score (0-1) 
                    for volunteer v to be in room r at hour h
        pref_weight: Weight for the preference satisfaction objective
        change_weight: Weight for the room change minimization objective
        
    Returns:
        schedule: 3D binary array of shape (n_volunteers, n_rooms, n_hours) representing the assignments
        model: Z3 solver model if a solution is found, None otherwise
    """
    # Initialize the Z3 optimizer
    optimizer = Optimize()
    
    # Define the decision variables - x[v, r, h] = True if volunteer v is assigned to room r at hour h
    x = {}
    for v in range(n_volunteers):
        for r in range(n_rooms):
            for h in range(n_hours):
                x[v, r, h] = Bool(f"x_{v}_{r}_{h}")
    
    # ----- CONSTRAINTS -----
    
    # Constraint 1: Each volunteer can be assigned to at most one room at any hour
    for v in range(n_volunteers):
        for h in range(n_hours):
            rooms_for_vol_at_hour = [x[v, r, h] for r in range(n_rooms)]
            optimizer.add(AtMost(*rooms_for_vol_at_hour, 1))
    
    # Constraint 2: Volunteers can only be assigned during their available hours
    for v in range(n_volunteers):
        for h in range(n_hours):
            if availability[v][h] == 0:  # If volunteer is not available
                for r in range(n_rooms):
                    optimizer.add(Not(x[v, r, h]))
    
    # Constraint 3: Each room must have at least one volunteer at every hour
    for r in range(n_rooms):
        for h in range(n_hours):
            vols_for_room_at_hour = [x[v, r, h] for v in range(n_volunteers)]
            optimizer.add(AtLeast(*vols_for_room_at_hour, 1))
    
    # ----- OBJECTIVE FUNCTIONS -----
    
    # Objective 1: Maximize preference satisfaction
    preference_terms = []
    for v in range(n_volunteers):
        for r in range(n_rooms):
            for h in range(n_hours):
                # Scale preferences to integers (0-100) for Z3
                pref_score = int(preferences[v][r][h] * 100)
                preference_terms.append(If(x[v, r, h], pref_score, 0))
    
    preference_sum = Sum(preference_terms)
    
    # Objective 2: Minimize room changes
    change_vars = []
    
    for v in range(n_volunteers):
        for h in range(n_hours - 1):
            # Track if volunteer is active in consecutive hours
            active_h = Or([x[v, r, h] for r in range(n_rooms)])
            active_h_next = Or([x[v, r, h+1] for r in range(n_rooms)])
            active_both = And(active_h, active_h_next)
            
            # For each room, check if volunteer stays in the same room
            same_room_indicators = []
            for r in range(n_rooms):
                same_room = And(x[v, r, h], x[v, r, h+1])
                same_room_indicators.append(same_room)
            
            # If volunteer is active in both hours but not in same room, it's a change
            same_room_any = Or(same_room_indicators)
            change = And(active_both, Not(same_room_any))
            change_vars.append(If(change, 1, 0))
    
    change_count = Sum(change_vars)
    
    # ----- OPTIMIZATION -----
    
    # Create a weighted objective function
    # Since we're maximizing preference_sum and minimizing change_count,
    # we use positive weight for preference and negative weight for changes
    
    # Scale the objectives appropriately
    # Assuming max preference_sum could be n_volunteers * n_rooms * n_hours * 100
    max_pref = n_volunteers * n_rooms * n_hours * 100
    # Assuming max change_count could be n_volunteers * (n_hours - 1)
    max_changes = n_volunteers * (n_hours - 1)
    
    # Normalize both objectives to similar scales and create weighted sum
    # Note: preference_sum is positive (maximizing) and change_count is negative (minimizing)
    combined_objective = pref_weight * (preference_sum / max_pref) - change_weight * (change_count / max_changes)
    
    # Add the combined objective to the optimizer
    optimizer.maximize(combined_objective)
    
    # ----- SOLVING AND RESULTS -----
    
    # Check if the problem is satisfiable
    result = optimizer.check()
    if result == sat:
        model = optimizer.model()
        # Create a numpy array to represent the solution
        schedule = np.zeros((n_volunteers, n_rooms, n_hours), dtype=int)
        
        # Extract solution from model
        for v in range(n_volunteers):
            for r in range(n_rooms):
                for h in range(n_hours):
                    if is_true(model.evaluate(x[v, r, h])):
                        schedule[v, r, h] = 1
        
        return schedule, model
    else:
        return None, None

def adapt_solution_format(schedule, n_volunteers, n_rooms, n_hours):
    """
    Converts our 3D array schedule to a dictionary format where
    keys are (person, hour) tuples and values are room numbers.
    
    Args:
        schedule: 3D binary array representing assignments
        n_volunteers, n_rooms, n_hours: Problem dimensions
        
    Returns:
        solution_dict: Dictionary mapping (person, hour) to room
    """
    solution_dict = {}
    
    for v in range(n_volunteers):
        for h in range(n_hours):
            for r in range(n_rooms):
                if schedule[v, r, h] == 1:
                    # In the display_schedule format, people are 1-indexed
                    # but we'll keep 0-indexed for consistency
                    solution_dict[(v, h)] = r
    
    return solution_dict

def display_schedule(solution, n_rooms, n_hours, n_people, preferences=None):
    """
    Display the surveillance schedule in a readable format.
    
    Parameters:
    - solution: Dictionary mapping (person, hour) to room
    - n_rooms: Number of rooms
    - n_hours: Number of hours
    - n_people: Number of people
    - preferences: Optional preferences array for calculating satisfaction
    """
    # Display the schedule
    print("Surveillance Schedule:")
    print("---------------------")
    print("Hour | " + " | ".join([f"Room {r+1}" for r in range(n_rooms)]))
    print("-" * (6 + n_rooms * 10))
    
    for h in range(n_hours):
        room_assignments = ["" for _ in range(n_rooms)]
        for p in range(n_people):
            if (p, h) in solution:
                r = solution[(p, h)]
                room_assignments[r] += f"V{p+1} "  # Changed P to V for Volunteer
        
        print(f"{h+1:4d} | " + " | ".join([f"{assignment:8s}" for assignment in room_assignments]))
    
    # Count room changes
    changes = 0
    for p in range(n_people):
        for h in range(1, n_hours):
            if (p, h-1) in solution and (p, h) in solution:
                if solution[(p, h-1)] != solution[(p, h)]:
                    changes += 1
    
    print(f"\nTotal room changes: {changes}")
    
    # Calculate preference satisfaction if preferences are provided
    if preferences is not None:
        total_preference = 0
        for p in range(n_people):
            for h in range(n_hours):
                if (p, h) in solution:
                    r = solution[(p, h)]
                    total_preference += preferences[p][r][h]
        
        print(f"Total preference satisfaction: {total_preference}")
        
    return changes, total_preference if preferences is not None else None

def print_schedule(schedule, n_volunteers, n_rooms, n_hours):
    """
    Print a human-readable schedule.
    
    Args:
        schedule: 3D binary array representing assignments
        n_volunteers, n_rooms, n_hours: Problem dimensions
    """
    print("\nDetailed Schedule:")
    print("-" * 50)
    
    # Print column headers (hours)
    print("Room  |", end="")
    for h in range(n_hours):
        print(f" Hour {h} |", end="")
    print()
    print("-" * (7 + 9 * n_hours))
    
    # Print room assignments
    for r in range(n_rooms):
        print(f"Room {r} |", end="")
        for h in range(n_hours):
            vols = [v for v in range(n_volunteers) if schedule[v, r, h] == 1]
            vols_str = ", ".join(map(str, vols)) if vols else "None"
            print(f" {vols_str:<7}|", end="")
        print()
    
    print("-" * (7 + 9 * n_hours))
    
    # Print individual volunteer schedules
    print("\nVolunteer Schedules:")
    print("-" * 50)
    
    for v in range(n_volunteers):
        print(f"Volunteer {v}: ", end="")
        assignments = []
        for h in range(n_hours):
            for r in range(n_rooms):
                if schedule[v, r, h] == 1:
                    assignments.append(f"Hour {h} in Room {r}")
        
        if assignments:
            print(", ".join(assignments))
        else:
            print("No assignments")

def example2():
    """
    Example usage of the surveillance scheduling system with a larger scenario.
    """
    # Example parameters
    n_rooms = 7
    n_hours = 10
    n_volunteers = 30
    
    # Generate random preferences for example
    np.random.seed(42)  # For reproducibility
    
    # Generate random preferences (0-3 scale)
    raw_preferences = np.random.randint(0, 3, size=(n_volunteers, n_rooms, n_hours))
    
    # Normalize to 0-1 scale for our implementation
    preferences = raw_preferences / 2.0
    
    # Make room 6 highly preferred for all volunteers (as in the example code)
    for v in range(n_volunteers):
        preferences[v, 6, :] = 1.0  # Set to maximum preference
    
    # Generate availability (at random)
    availability = np.random.randint(0, 2, size=(n_volunteers, n_hours))

    
    # Print some preference info
    print("Preference Summary:")
    #print(f"- Volunteers with high preference for Room 6: All")
    print(f"- Average preference value: {np.mean(preferences):.2f}")
    print()
    
    # Set weights for objectives (change penalty of 5 corresponds to higher weight for change minimization)
    pref_weight = 0.5
    change_weight = 0.5
    print(f"Using preference weight: {pref_weight}")
    print(f"Using change weight: {change_weight}")
    
    print("Solving surveillance scheduling problem...")
    print(f"- {n_rooms} rooms")
    print(f"- {n_hours} hours")
    print(f"- {n_volunteers} volunteers")

    # Print preference matrix for reference
    print("(Raw) Preference Values (Person, Room, Hour):")
    print("-" * 50)
    print(" " * 4 + "Hour:  " + "".join((f"{h+1} " for h in range(n_hours))))
    print("-" * 50)
    for p in range(n_volunteers):
        print(f"Volunteer {p+1}:")
        for r in range(n_rooms):
            print(f"  Room {r+1}: {raw_preferences[p][r]}")
    print()
    print("-" * 50)
    
    # Solve the problem
    schedule, model = surveillance_scheduling(
        n_rooms, n_hours, n_volunteers, availability, preferences, 
        pref_weight=pref_weight, change_weight=change_weight
    )
    
    if schedule is not None:
        # Convert to the format expected by display_schedule
        solution_dict = adapt_solution_format(schedule, n_volunteers, n_rooms, n_hours)
        
        # Display using the provided function
        display_schedule(solution_dict, n_rooms, n_hours, n_volunteers, raw_preferences)
        
        # Additional statistics
        pref_value, changes = calculate_metrics(schedule, preferences, n_volunteers, n_rooms, n_hours)
        
        print("\nAdditional Statistics:")
        print(f"- Total normalized preference satisfaction: {pref_value:.2f}")
        
        # Print summary of room assignments
        print("\nRoom Assignment Summary:")
        for r in range(n_rooms):
            assignments = np.sum(schedule[:, r, :])
            print(f"Room {r+1}: {assignments} total assignments across all hours")
        
        # This is a list of volunteers with consistent room assignments
        # (i.e., they are assigned to at most one room at any hour)
        # A volunteer is assigned to at most one room at any hour if 
        consistent_vols = [
            (v, schedule[v, :, :]) for v in range(n_volunteers)
            if len(
                list(
                    r for r in range(n_rooms) for h in range(n_hours) 
                    if schedule[v, r, h] == 1
                )
            ) <= 1
        ]
        print(f"\nVolunteers with consistent room assignments: {consistent_vols}")
        print(f" Consistent room assignments:\n{schedule[[v for v, _ in consistent_vols], :, :]}")

        # This is a list of tuples (volunteer, hour) where the volunteer has more than one room assignment
        # at that hour
        conflicting_assignement = [
            (volunteer, hour) 
            for volunteer in range(n_volunteers) 
            for hour in range(n_hours) 
            if np.sum(schedule[volunteer, :, hour]) > 1
        ]
        print(f"\nConflicting volunteers (volunteer, hour): {conflicting_assignement}")

        # Check availability satisfaction
        # is_assined[v][h] = np.sum(schedule[v, :, h])
        is_assigned = [
            [np.sum(schedule[v, :, h]) for h in range(n_hours)] 
            for v in range(n_volunteers)
        ]
        print(f"Unsatisfied availability constrains: {np.sum(availability * is_assigned - is_assigned)}")
    else:
        print("Failed to find a solution.")

import time

start_time = time.time()

example2()

end_time = time.time()

print(f"Time taken: {end_time - start_time:.4f} seconds")
