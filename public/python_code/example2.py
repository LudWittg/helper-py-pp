from z3 import *
import numpy as np
import time

def solve_room_surveillance(
    num_rooms, num_hours, num_people, 
    availability, preferences, is_stage_manager, room_requirements,
    weight=1.0
):
    """
    Solves the room surveillance optimization problem using Z3 SMT solver.
    """
    # Create Z3 optimizer
    opt = Optimize()
    
    # Define languages
    languages = ["English", "Italian"]
    
    # Define decision variables
    # x[p][r][t] = True if person p is assigned to room r at time t
    x = {(p, r, t): Bool(f"x_{p}_{r}_{t}") 
         for p in range(num_people) 
         for r in range(num_rooms) 
         for t in range(num_hours)}
    
    # same[p][t] = True if person p stays in the same room from time t-1 to time t
    same = {(p, t): Bool(f"same_{p}_{t}") 
            for p in range(num_people) 
            for t in range(1, num_hours)}
    
    # Define constraints
    
    # 1. Availability constraint
    for p in range(num_people):
        for r in range(num_rooms):
            for t in range(num_hours):
                if (p, t) in availability and not availability[(p, t)]:
                    opt.add(Not(x[(p, r, t)]))
    
    # 2. Single assignment constraint: each person can be assigned to at most one room
    for p in range(num_people):
        for t in range(num_hours):
            # At most one room per person per time
            opt.add(PbLe([(x[(p, r, t)], 1) for r in range(num_rooms)], 1))
    
    # 3. Room coverage constraint: each room must have at least one person
    for r in range(num_rooms):
        for t in range(num_hours):
            # At least one person per room per time
            opt.add(PbGe([(x[(p, r, t)], 1) for p in range(num_people)], 1))
    
    # 4. Language requirement constraint
    for r in range(num_rooms):
        for t in range(num_hours):
            for l, lang in enumerate(languages):
                if (r, t, lang) in room_requirements and room_requirements[(r, t, lang)]:
                    # For each required language, at least one person who is a stage manager 
                    # for that language must be assigned
                    stage_manager_assignments = []
                    for p in range(num_people):
                        if (p, lang) in is_stage_manager and is_stage_manager[(p, lang)]:
                            stage_manager_assignments.append((x[(p, r, t)], 1))
                    
                    if stage_manager_assignments:  # Only add constraint if there are capable stage managers
                        opt.add(PbGe(stage_manager_assignments, 1))
    
    # 5. Define same-room variables
    for p in range(num_people):
        for t in range(1, num_hours):
            # The following constraints define same[p, t] to be true if 
            # and only if person p is assigned to the same room at time t-1 and t
            
            # For each room, create a variable that's true if person p is in that room at both t-1 and t
            room_same = [And(x[(p, r, t-1)], x[(p, r, t)]) for r in range(num_rooms)]
            
            # same[p, t] is true if any of the room_same variables is true
            opt.add(same[(p, t)] == Or(room_same))
    
    # Define objective function
    
    # Preferences term: sum of preference scores for assigned positions
    preferences_term = []
    for p in range(num_people):
        for r in range(num_rooms):
            for t in range(num_hours):
                if (p, r, t) in preferences:
                    # Multiply preference by 100 to convert to integers for Z3
                    preferences_term.append((x[(p, r, t)], int(preferences[(p, r, t)] * 100)))
    
    # Room continuity term: reward for staying in the same room
    continuity_term = []
    for p in range(num_people):
        for t in range(1, num_hours):
            continuity_term.append((same[(p, t)], int(weight * 100)))
    
    # Combine terms
    objective = preferences_term + continuity_term
    
    # Set objective function to maximize
    h = opt.maximize(Sum([If(cond, weight, 0) for cond, weight in objective]))
    
    # Check if the problem is satisfiable
    if opt.check() == sat:
        model = opt.model()
        
        # Extract assignments
        assignments = {}
        for p in range(num_people):
            for t in range(num_hours):
                assigned_room = None
                for r in range(num_rooms):
                    if is_true(model.evaluate(x[(p, r, t)])):
                        assigned_room = r
                        break
                
                if assigned_room is not None:
                    assignments[(p, t)] = assigned_room
        
        # Also extract which stage managers are assigned to each room
        stage_manager_assignments = {}
        for r in range(num_rooms):
            for t in range(num_hours):
                stage_manager_assignments[(r, t)] = {"English": [], "Italian": []}
                for p in range(num_people):
                    if is_true(model.evaluate(x[(p, r, t)])):
                        for lang in ["English", "Italian"]:
                            if (p, lang) in is_stage_manager and is_stage_manager[(p, lang)]:
                                stage_manager_assignments[(r, t)][lang].append(p)
        
        # Calculate objective value
        obj_value = opt.lower(h).as_long() / 100.0  # Convert back from integers
        
        return model, assignments, stage_manager_assignments, obj_value
    else:
        return None, None, None, None

def print_schedule(assignments, stage_manager_assignments, num_people, num_rooms, num_hours):
    """Prints the room assignment schedule in a readable format."""
    print("\nSchedule:")
    print("-" * 80)
    print("Time | " + " | ".join([f"Room {r}" for r in range(num_rooms)]))
    print("-" * 80)
    
    for t in range(num_hours):
        # For each room, collect all assigned people
        room_assignments = ["" for _ in range(num_rooms)]
        
        for p in range(num_people):
            if (p, t) in assignments:
                r = assignments[(p, t)]
                room_assignments[r] += f"P{p} "
        
        # Print the basic schedule
        print(f" {t:2d} | " + " | ".join([f"{a:<15}" for a in room_assignments]))
        
        # Print stage manager information
        stage_mgr_info = ["" for _ in range(num_rooms)]
        for r in range(num_rooms):
            if (r, t) in stage_manager_assignments:
                eng = stage_manager_assignments[(r, t)]["English"]
                ita = stage_manager_assignments[(r, t)]["Italian"]
                
                stage_mgr_info[r] = f"E:[{','.join(f'P{p}' for p in eng)}] "
                stage_mgr_info[r] += f"I:[{','.join(f'P{p}' for p in ita)}]"
        
        print(f"    | " + " | ".join([f"{a:<15}" for a in stage_mgr_info]))
        print("-" * 80)

def simple_example():
    """A simple, satisfiable example of the room surveillance problem."""
    # Problem parameters
    num_rooms = 2
    num_hours = 3
    num_people = 4
    languages = ["English", "Italian"]
    
    # =========================================================================
    # Define the problem instance
    # =========================================================================
    
    # Availability: Who's available when
    # All people are available at all times except:
    # - Person 0 is unavailable at hour 2
    # - Person 3 is unavailable at hour 0
    availability = {(p, t): True for p in range(num_people) for t in range(num_hours)}
    availability[(0, 2)] = False  # Person 0 is unavailable at hour 2
    availability[(3, 0)] = False  # Person 3 is unavailable at hour 0
    
    # Stage manager capabilities:
    # - Person 0: English stage manager
    # - Person 1: Italian stage manager
    # - Person 2: Both English and Italian stage manager
    # - Person 3: Not a stage manager
    is_stage_manager = {
        (0, "English"): True,  (0, "Italian"): False,
        (1, "English"): False, (1, "Italian"): True,
        (2, "English"): True,  (2, "Italian"): True,
        (3, "English"): False, (3, "Italian"): False
    }
    
    # Room language requirements:
    # - Room 0 needs English at hours 0 and 2
    # - Room 1 needs Italian at hours 1 and 2
    # - Room 0 needs Italian at hour 1
    # - Room 1 needs English at hour 0
    room_requirements = {
        (0, 0, "English"): True,  (0, 0, "Italian"): False,
        (0, 1, "English"): False, (0, 1, "Italian"): True,
        (0, 2, "English"): True,  (0, 2, "Italian"): False,
        (1, 0, "English"): True,  (1, 0, "Italian"): False,
        (1, 1, "English"): False, (1, 1, "Italian"): True,
        (1, 2, "English"): False, (1, 2, "Italian"): True
    }
    
    # Preferences: Each volunteer has some room preferences
    # Higher values indicate stronger preferences
    preferences = {}
    
    # Person 0 prefers Room 0
    preferences[(0, 0, 0)] = 8.0  # Person 0, Room 0, Hour 0
    preferences[(0, 0, 1)] = 8.0  # Person 0, Room 0, Hour 1
    preferences[(0, 1, 0)] = 3.0  # Person 0, Room 1, Hour 0
    preferences[(0, 1, 1)] = 3.0  # Person 0, Room 1, Hour 1
    
    # Person 1 prefers Room 1
    preferences[(1, 0, 0)] = 2.0  # Person 1, Room 0, Hour 0
    preferences[(1, 0, 1)] = 2.0  # Person 1, Room 0, Hour 1
    preferences[(1, 0, 2)] = 2.0  # Person 1, Room 0, Hour 2
    preferences[(1, 1, 0)] = 7.0  # Person 1, Room 1, Hour 0
    preferences[(1, 1, 1)] = 7.0  # Person 1, Room 1, Hour 1
    preferences[(1, 1, 2)] = 7.0  # Person 1, Room 1, Hour 2
    
    # Person 2 has no strong preferences
    for r in range(num_rooms):
        for t in range(num_hours):
            preferences[(2, r, t)] = 5.0  # Neutral preference
    
    # Person 3 prefers Room 0 at Hour 1 and Room 1 at Hour 2
    preferences[(3, 0, 1)] = 9.0  # Person 3, Room 0, Hour 1
    preferences[(3, 0, 2)] = 4.0  # Person 3, Room 0, Hour 2
    preferences[(3, 1, 1)] = 3.0  # Person 3, Room 1, Hour 1
    preferences[(3, 1, 2)] = 9.0  # Person 3, Room 1, Hour 2
    
    # Fill in any missing preferences with a default value
    for p in range(num_people):
        for r in range(num_rooms):
            for t in range(num_hours):
                if (p, r, t) not in preferences:
                    preferences[(p, r, t)] = 1.0  # Default low preference
    
    # =========================================================================
    # Solve the problem
    # =========================================================================
    
    # We'll prioritize room continuity moderately
    weight = 3.0  # Weight for room continuity vs preferences
    
    print("Simple Room Surveillance Example:")
    print(f"- {num_rooms} rooms, {num_hours} hours, {num_people} people")
    print("\nAvailability:")
    for p in range(num_people):
        avail_hours = [t for t in range(num_hours) if availability[(p, t)]]
        print(f"Person {p}: Hours {avail_hours}")
    
    print("\nStage Manager Capabilities:")
    for p in range(num_people):
        langs = []
        if is_stage_manager.get((p, "English"), False):
            langs.append("English")
        if is_stage_manager.get((p, "Italian"), False):
            langs.append("Italian")
        if not langs:
            print(f"Person {p}: Not a stage manager")
        else:
            print(f"Person {p}: Stage manager for {', '.join(langs)}")
    
    print("\nRoom Language Requirements:")
    for r in range(num_rooms):
        for t in range(num_hours):
            req_langs = []
            if room_requirements.get((r, t, "English"), False):
                req_langs.append("English")
            if room_requirements.get((r, t, "Italian"), False):
                req_langs.append("Italian")
            if req_langs:
                print(f"Room {r}, Hour {t}: Requires {', '.join(req_langs)}")
    
    # Solve the problem
    model, assignments, stage_manager_assignments, obj_value = solve_room_surveillance(
        num_rooms, num_hours, num_people, 
        availability, preferences, is_stage_manager, room_requirements,
        weight
    )
    
    if model is not None:
        print("\nSolution found!")
        print(f"Objective value: {obj_value:.2f}")
        print_schedule(assignments, stage_manager_assignments, num_people, num_rooms, num_hours)
        
        # Count room changes
        room_changes = 0
        for p in range(num_people):
            for t in range(1, num_hours):
                if (p, t-1) in assignments and (p, t) in assignments:
                    if assignments[(p, t-1)] != assignments[(p, t)]:
                        room_changes += 1
                        print(f"Person {p} changed from Room {assignments[(p, t-1)]} to Room {assignments[(p, t)]} between hours {t-1} and {t}")
        
        print(f"\nTotal room changes: {room_changes}")
        
        # Verify language requirements are met
        language_requirements_met = True
        for r in range(num_rooms):
            for t in range(num_hours):
                for lang in ["English", "Italian"]:
                    if room_requirements.get((r, t, lang), False):
                        if not stage_manager_assignments[(r, t)][lang]:
                            print(f"Warning: Room {r} at time {t} requires {lang} but no stage manager is assigned!")
                            language_requirements_met = False
        
        if language_requirements_met:
            print("All language requirements are satisfied!")
    else:
        print("No solution found. The problem might be infeasible.")

if __name__ == "__main__":
    start_time = time.time()

    simple_example()

    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.4f} seconds")
