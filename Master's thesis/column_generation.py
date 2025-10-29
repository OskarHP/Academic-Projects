from Instance import Instance  # Classes
from CG_Master_Sub import Master, Sub  # Classes 
from CG_store_solution import *  # Functions to store solution
from CG_generate_initial_routes import *  # Functions to generate initial solution to the MP
from parameters import *  # Parameter values

from gurobipy import GRB  
import time
from joblib import Parallel, delayed

"""
The core of the column generation algorithm for the VRP. Changeable parameters: n_vehicles, n_depots, n_customers,
n_stations and set_seed. Writes the instance to instance.txt, solves it, plots the solution and writes a summary
in CG_solution.txt
"""


def solve_pricing(sub):  # To include parallelization
    sub.model.optimize()
    return sub


def solve_column_generation(instance):
    """
    The core function of the column generation algorithm. Solves the input instance and writes the solution.
    :param instance:
    :return: The solution
    """
    print("Initializing routes to solve RMP:")
    routes = generate_initial_routes(instance)  
    
    print("Starting column generation:")
    find_routes = True  # False when column generation procedure is finished
    fine_tuning = False  # True later in the search
    prev_obj = float('inf')

    # To store convergence data
    RMP_upper = []
    RMP_lower = []
    iteration = 0
    
    while find_routes:
        iteration += 1
        # Create a new Master object and solve its model to optimality
        master = Master(instance, routes, 0)
        master.model.optimize()
        curr_obj = master.model.objVal

        # We start by just finding A route with negative reduced cost, instead of the best one
        if not fine_tuning and curr_obj > (1 - tuning_param[0]) * prev_obj and curr_obj != 10**8:  # 10^8 is the "dummy column" cost
            params['BestObjStop'] = 'default'
            print("Fine tuning started")
            fine_tuning = True

        if fine_tuning and (RMP_upper[-1] - RMP_lower[-1]) / RMP_lower[-1] <= tuning_param[1]:
            print("Column generation ended due to lack of progress.")
            break

        print("Current objective value: %.0f" % curr_obj)
        RMP_upper.append(curr_obj)
        duals = np.array([constraint.pi for constraint in master.model.getConstrs()])  # Duals to RMP constraints
        subproblems = [None] * len(instance.vehicles)  # One subproblem for each vehicle
        
        # For each vehicle, we create a subproblem, solve it and if its objective value is negative we add the best route
        for k in instance.vehicles:
            sub = Sub(instance, k, duals, params, 0)
            subproblems[k - 1] = sub

        # Parallel optimization of subproblems
        time_before_opt = time.time()
        subproblems = Parallel(n_jobs=-1, prefer="threads")(delayed(solve_pricing)(sub) for sub in subproblems)
        print("The parallelization to optimize all subproblems in iteration %d took %.2f seconds." %
              (iteration, time.time() - time_before_opt))
        RMP_lb = RMP_upper[-1]  # To calculate a lower bound for the RMP in each iteration
        
        # Gather multiple routes that have negative reduced cost for each subproblem
        for k in instance.vehicles:
            sub = subproblems[k - 1]
            solution_number = 0
            sub.model.setParam("SolutionNumber", solution_number)
            while solution_number < sub.model.SolCount and sub.model.PoolObjval < 0:
                vehicle_route_idx = master.number_of_routes[k - 1] + 1  # The route index for that specific vehicle
                routes[k, vehicle_route_idx] = get_route(instance, sub)
                RMP_lb += sub.model.PoolObjval  # We add the reduced cost for the added route
                # Update the route indices
                master.number_of_routes[k - 1] += 1
                solution_number += 1
                sub.model.setParam("SolutionNumber", solution_number)
        RMP_lower.append(max(0, RMP_lb))  # Add the lower bound, we KNOW that a lower bound
        # If we cannot find any improving routes for any vehicle we stop the search
        if all(sub.model.objVal > -1e-5 for sub in subproblems):
            find_routes = False
        prev_obj = curr_obj

    # Make a final optimization
    master = Master(instance, routes, 0)
    print("Column generation finished, found %d routes." % sum(master.number_of_routes))
    master.model.optimize()
    relaxed_optimal = master.model.getObjective().getValue()
    print("Optimal objective value for relaxed RMP: %.0f" % relaxed_optimal)
    print("Set variables to INTEGER and optimize:")
    for var in master.lambda_.values():
        var.vtype = GRB.INTEGER
    master.model.update()
    final_opt = time.time()
    master.model.optimize()
    print("Obtained objective value: %.0f" % master.model.getObjective().getValue())
    print("Final optimization of RMP took %.2f seconds" % (time.time() - final_opt))
    return master, relaxed_optimal


def get_route(instance, sub):
    """
    Get the route that yields the lowest reduced cost
    :param instance: The chosen instance of customers and vehicles
    :param sub: The specific subproblem (for the vehicle)
    :return: A dictionary containing nodes loading and unloading, total time and cost for the route
    and the route as a list together with its route index.
    """
    route = [0]
    next_stop = find_next_stop(instance, sub, 0)
    while next_stop != 0 and next_stop != instance.fic_depot_idx:
        route.append(next_stop)
        current_stop = next_stop
        next_stop = find_next_stop(instance, sub, current_stop)

    route.append(next_stop)
    summary = {'Loading': {i: sum(sub.m_L[i, h].xn for h in instance.loading_types) for i in instance.nodes},
               'Unloading': {i: sum(sub.m_U[i, h].xn for h in instance.loading_types) for i in instance.nodes},
               'Route_time': sub.compute_time(instance),
               'Route_cost': sub.compute_cost(instance),
               'Charging': {i: sub.p[i].xn for i in instance.nodes},
               'Route': route
               }
    return summary


def find_next_stop(instance, sub, current):
    """
    Finds the next stop for the route given the current stop
    :param instance: The problem instance
    :param current: The current stop
    :param sub: The subproblem
    :return: Index of the next stop
    """
    for j in instance.nodes:
        if (current, j) in instance.arcs and sub.x[current, j].xn > 0.5:
            return j


if __name__ == '__main__':
    # Main function, performs column generation and produces relevant results
    computation_times = []
    obtained_costs = []
    optimal_relaxed = []

    for n_customers in range(n_min_customers, n_max_customers + 1):
        start_time = time.time()
        insta = Instance(n_vehicles, n_depots, n_customers, n_stations_at_customers, n_pure_charging,
                         set_seed, n_remove_arcs=0)
        insta.write_instance_to_file('CG_C%dS%d-V%d.txt' % (n_customers, n_pure_charging, n_vehicles))
        m1, rel_optimal = solve_column_generation(insta)  # The Master problem with optimized variables
        if m1.model.getObjective().getValue() == 10**8:
            print("The problem is infeasible. Larger instances will also be infeasible. Ending program...")
            break
        end_time = time.time()
        print("The column generation algorithm took %.2f seconds." % (end_time - start_time))

        # Plot the solution
        plot_solution(insta, m1.lambda_, m1.routes)
        write_solution_to_file(insta, m1.lambda_, m1.routes,
                               'CG_C%dS%d-V%d_sol.txt' % (n_customers, n_pure_charging, n_vehicles))
        computation_times.append(end_time - start_time)
        obtained_costs.append(m1.model.getObjective().getValue())
        optimal_relaxed.append(rel_optimal)
    plt.show()
