import pandas as pd
import numpy as np
import random
from operator import itemgetter
import math
import operator

instance_path = './instance/i1.txt'
route_probs = {}
result_file = open("res-5m-i1.txt", "w")


def load_data(instance_path):
    instance = open(instance_path, 'r')
    lines = instance.readlines()
    second_line = lines[2].strip().split()
    num_vehicles, capacity = map(int, second_line)

    # dohvati koordinate skladišta i najkasnije vrijeme povratka u skladište
    depo_line = lines[7].strip().split()
    depo_line = list(map(int, depo_line))
    depo_line = np.array(depo_line)

    depo_id, depo_x, depo_y, depo_demand, depo_ready_time, depo_due_time, depo_service_time = map(int, depo_line)

    customers = pd.read_csv(instance_path, delim_whitespace=True, skiprows=[0, 1, 2, 3, 4, 5, 6, 7],
                            names=["ID", "XCOORD", "YCOORD", "DEMAND", "READY_TIME", "DUE_DATE", "SERVICE_TIME"])

    customers = customers.to_numpy()
    num_customers = len(customers)

    return num_vehicles, capacity, depo_x, depo_y, depo_due_time, customers, depo_line, num_customers


class Depo:
    def __init__(self, id, x_coord, y_coord, demand, ready_time, due_date, service_time):
        self.id = id
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.demand = demand
        self.ready_time = ready_time
        self.due_date = due_date
        self.service_time = service_time

    def __str__(self):
        return '[{self.id},{self.x_coord},{self.y_coord},{self.demand},{self.ready_time},{self.due_date},{self.service_time}]'.format(
            self=self)


class Customer:
    def __init__(self, id=0, x_coord=0, y_coord=0, demand=0, ready_time=0, due_date=0, service_time=0):
        self.id = id
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.demand = demand
        self.ready_time = ready_time
        self.due_date = due_date
        self.service_time = service_time

    def __str__(self):
        return '[{self.id},{self.x_coord},{self.y_coord},{self.demand},{self.ready_time},{self.due_date},{self.service_time}]'.format(
            self=self)


class Route:

    def __init__(self):
        self.list_of_customers = []
        self.current_demand = 0
        self.time_spent = 0
        self.distance_to = 0
        self.list_of_distances = []
        self.list_of_service_starts = []
        self.overall_distance = 0

    def __len__(self):
        return len(self.list_of_customers)

    def __str__(self):
        ispis = ''
        for customer in self.list_of_customers:
            ispis = ispis + str(customer) + ','

        ispis = ispis[:-1]
        return '[{ispis}]'.format(ispis=ispis)

    def update_time_spent(self, time_spent):
        self.time_spent = time_spent

    def update_service_starts(self, service_start):
        self.list_of_service_starts.append(service_start)

    def update_distance_to(self, distance_to):
        self.distance_to = distance_to

    def add_to_route(self, customer):
        self.list_of_customers.append(customer)

    def update_current_demand(self, demand):
        self.current_demand = demand

    def pretty_route_str(self):
        pretty = ''
        for i, customer in enumerate(self.list_of_customers):
            if i > len(self.list_of_service_starts) - 1:
                continue
            pretty += '{customer_id}({udaljenost})->'.format(customer_id=customer.id,
                                                             udaljenost=math.ceil(self.list_of_service_starts[i]))
        pretty = pretty[:-2]
        return pretty


def get_solution_string(s_best, i):
    s_best_string = ""

    if i != -1:
        s_best_string = "\niter: " + str(i + 1) + "\n"
    s_best_string += str(len(s_best)) + "\n"
    s_best_distance = 0
    for j, r in enumerate(s_best):
        r_string = r.pretty_route_str()
        s_best_string += str(j + 1) + ": " + r_string + "\n"
        s_best_distance += r.overall_distance
    s_best_string += str(s_best_distance)

    return s_best_string


def calculate_total_distance(routes):
    total_distance = 0
    for route in routes:
        total_distance += route.overall_distance

    return total_distance


def remove_empty_routes(potential_neighbor):
    for route in potential_neighbor:
        if len(route) == 2:
            potential_neighbor.remove(route)

    return potential_neighbor


def check_potential_route(list_of_customers):
    feasible = True

    route = Route()
    list_of_customers = list_of_customers[1:-1]

    route.add_to_route(depo)
    route.update_service_starts(0)

    for customer in list_of_customers:
        last_customer = route.list_of_customers[-1]

        updated_demand = route.current_demand + customer.demand
        euclidean_distance = (math.sqrt(
            (last_customer.x_coord - customer.x_coord) ** 2 + (last_customer.y_coord - customer.y_coord) ** 2))
        arrival_time = route.time_spent + math.ceil(euclidean_distance)

        if arrival_time > customer.due_date:
            feasible = False
            return route, feasible

        if arrival_time < customer.ready_time:
            arrival_time = customer.ready_time

        finish_time = arrival_time + customer.service_time
        euclidean_distance_to_depo = math.sqrt(
            (depo.x_coord - customer.x_coord) ** 2 + (depo.y_coord - customer.y_coord) ** 2)
        return_time = finish_time + euclidean_distance_to_depo

        if return_time > depo.due_date:
            feasible = False
            return route, feasible

        if updated_demand > capacity:
            feasible = False
            return route, feasible

        route.add_to_route(customer)
        route.update_current_demand(updated_demand)
        route.update_time_spent(finish_time)
        route.update_service_starts(arrival_time)

        route.overall_distance += euclidean_distance

    last_customer = route.list_of_customers[-1]
    euclidean_distance_to_depo = math.sqrt(
        (depo.x_coord - last_customer.x_coord) ** 2 + (depo.y_coord - last_customer.y_coord) ** 2)
    arrival_time_at_depo = route.time_spent + euclidean_distance_to_depo
    route.update_service_starts(arrival_time_at_depo)
    route.overall_distance += euclidean_distance_to_depo

    route.add_to_route(depo)

    return route, feasible


def greedy_algorithm(num_vehicles, capacity, customers_data, depo_data, N):
    depo = Depo(*depo_data)

    customers = []
    for customer in customers_data:
        customers.append(Customer(*customer))

    routes = []
    customersCopymain = customers.copy()
    while len(routes) < num_vehicles:

        if len(customersCopymain) == 0:
            break

        route = Route()
        route.add_to_route(depo)
        route.update_service_starts(0)

        customersCopy = customersCopymain.copy()
        while True:

            if len(customersCopy) == 0:
                break

            last_customer = route.list_of_customers[-1]

            customer_dist_list = []
            for customer in customersCopy:
                euclidean_distance = math.sqrt(
                    (last_customer.x_coord - customer.x_coord) ** 2 + (last_customer.y_coord - customer.y_coord) ** 2)
                arrival_time = route.time_spent + euclidean_distance
                customer_dist_list.append((customer, arrival_time))
            sorted_customer_dist_list = sorted(customer_dist_list, key=itemgetter(1))

            sorted_customer_dist_list = [i[0] for i in sorted_customer_dist_list]
            if len(sorted_customer_dist_list) > N:
                sorted_customer_dist_list = sorted_customer_dist_list[:N]

            sorted_by_ready_time = sorted(sorted_customer_dist_list, key=operator.attrgetter('ready_time'))
            customer = sorted_by_ready_time[0]

            updated_demand = route.current_demand + customer.demand
            euclidean_distance = (math.sqrt(
                (last_customer.x_coord - customer.x_coord) ** 2 + (last_customer.y_coord - customer.y_coord) ** 2))
            arrival_time = route.time_spent + math.ceil(euclidean_distance)

            if arrival_time > customer.due_date:
                customersCopy.remove(customer)
                continue

            if arrival_time < customer.ready_time:
                arrival_time = customer.ready_time

            finish_time = arrival_time + customer.service_time
            euclidean_distance_to_depo = math.sqrt(
                (depo.x_coord - customer.x_coord) ** 2 + (depo.y_coord - customer.y_coord) ** 2)
            return_time = finish_time + euclidean_distance_to_depo

            if return_time > depo.due_date:
                customersCopy.remove(customer)
                continue

            if updated_demand > capacity:
                customersCopy.remove(customer)
                continue

            route.add_to_route(customer)
            route.update_current_demand(updated_demand)
            route.update_time_spent(finish_time)
            route.update_service_starts(arrival_time)

            route.overall_distance += euclidean_distance

            customersCopy.remove(customer)
            customersCopymain.remove(customer)

        # dodaj depo kao zadnje mjesto u ruti
        last_customer = route.list_of_customers[-1]
        euclidean_distance_to_depo = math.sqrt(
            (depo.x_coord - last_customer.x_coord) ** 2 + (depo.y_coord - last_customer.y_coord) ** 2)
        arrival_time_at_depo = route.time_spent + euclidean_distance_to_depo
        route.update_service_starts(arrival_time_at_depo)
        route.overall_distance += euclidean_distance_to_depo

        route.add_to_route(depo)
        routes.append(route)

    return routes


def generate_neighborhood_oropt(init_routes, zero_brojac, tabu_list):
    # ne koristena verzija generiranja susjedstva

    neighborhood = []
    changes = []

    shortest_route_sorted = sorted(init_routes, key=len)
    temp = 0
    while len(neighborhood) <= 100:
        shortest_route = shortest_route_sorted[temp]
        temp += 1
        i = init_routes.index(shortest_route)

        if len(shortest_route) > 4 and zero_brojac == 0:
            max_k = 3
        elif len(shortest_route) > 3 and zero_brojac < 2:
            max_k = 2
        else:
            max_k = 1

        for k in range(max_k, 0, -1):

            for j in range(1, len(shortest_route.list_of_customers) - max_k):

                customers_to_insert = shortest_route.list_of_customers[j:j + max_k]

                potential_neighbor = init_routes.copy()
                list_with_customer_to_insert = potential_neighbor[i].list_of_customers
                list_without_customer_to_insert = list_with_customer_to_insert.copy()
                for c in customers_to_insert:
                    list_without_customer_to_insert.remove(c)
                route_without_customer, feasible = check_potential_route(list_without_customer_to_insert)

                potential_neighbor[i] = route_without_customer
                potential_neighbor = remove_empty_routes(potential_neighbor)

                for k, route in enumerate(potential_neighbor):

                    for l, customer in enumerate(route.list_of_customers):

                        if l == 0 or l == len(route.list_of_customers) - 1:
                            continue

                        for cust in customers_to_insert:
                            if (cust, shortest_route.list_of_customers) in tabu_list:
                                continue

                        new_route = route.list_of_customers.copy()
                        potential_route = new_route[:l] + customers_to_insert + new_route[l:]

                        good_route, feasible = check_potential_route(potential_route)

                        if not feasible:
                            continue
                        else:
                            neighbor = potential_neighbor.copy()
                            neighbor[k] = good_route
                            neighborhood.append(neighbor)
                            for cust in customers_to_insert:
                                changes.append((cust, shortest_route.list_of_customers))

    return neighborhood, changes


def generate_neighborhood_changes(init_routes, tabu_list):
    # zamijene 1 za 1
    neighborhood = []
    changes = []

    chosen_routes = []

    while len(neighborhood) <= 10:
        random_route_1_ind = random.randint(0, len(init_routes) - 1)
        random_route_2_ind = random.randint(0, len(init_routes) - 1)

        random_route_1 = init_routes[random_route_1_ind]
        random_route_2 = init_routes[random_route_2_ind]

        while (random_route_1, random_route_2) in chosen_routes:
            random_route_1_ind = random.randint(0, len(init_routes) - 1)
            random_route_1 = init_routes[random_route_1_ind]
            random_route_2_ind = random.randint(0, len(init_routes) - 1)
            random_route_2 = init_routes[random_route_2_ind]

        if random_route_1 == random_route_2:
            continue

        chosen_routes.append((random_route_1, random_route_2))

        potential_neighbor = init_routes.copy()

        for i, customer1 in enumerate(random_route_1.list_of_customers):

            if customer1.id == 0:
                continue

            if (customer1, random_route_2) in tabu_list:
                continue

            for j, customer2 in enumerate(random_route_2.list_of_customers):
                new_route = init_routes.copy()
                if customer2.id == 0:
                    continue

                if (customer2, random_route_1) in tabu_list:
                    continue

                if j >= len(random_route_1.list_of_customers):
                    break
                n_route1 = new_route[random_route_1_ind].list_of_customers[:i] + [customer2] \
                           + new_route[random_route_1_ind].list_of_customers[i + 1:]
                n_route2 = new_route[random_route_2_ind].list_of_customers[:j] + [customer1] \
                           + new_route[random_route_2_ind].list_of_customers[j + 1:]

                good_route1, feasible1 = check_potential_route(n_route1)
                good_route2, feasible2 = check_potential_route(n_route2)

                if feasible1 and feasible2:
                    neighbor = potential_neighbor.copy()
                    neighbor[random_route_1_ind] = good_route1
                    neighbor[random_route_2_ind] = good_route2
                    neighborhood.append(neighbor)
                    changes.append((customer2, random_route_1.list_of_customers))
                    changes.append((customer1, random_route_2.list_of_customers))

    return neighborhood, changes


def generate_neighborhood_del_route(s, tabu_list, max_b):
    neighborhood = []
    changes = []

    initial_routes = s.copy()
    over_routes = []
    b = 0

    while len(neighborhood) <= 0 and b <= max_b:

        index_init_route = -1
        init_route = None

        init_route_tup = random.choice([k for k in route_probs for dummy in range(route_probs[k])])

        found = False
        for i, route in enumerate(initial_routes):
            if route.list_of_customers == list(init_route_tup):
                index_init_route = i
                init_route = route
                found = True
                break

        if found is False:
            continue

        route_probs[init_route_tup] -= 1

        over_routes.append(index_init_route)

        routes_without_random_route = initial_routes.copy()
        routes_without_random_route.remove(init_route)

        customeri = init_route.list_of_customers

        neighbor = routes_without_random_route.copy()
        nove_rute = routes_without_random_route.copy()
        route_feasible = True
        temp_changes = []

        for i, customer in enumerate(customeri):

            customer_inserted = False
            while True:
                route_feasible = True
                if customer.id == 0:
                    break

                for j, route in enumerate(nove_rute):

                    for k, cust in enumerate(route.list_of_customers):
                        if k == 0 or k == ((len(route.list_of_customers)) - 1):
                            continue
                        new_route = route.list_of_customers.copy()
                        potential_route = new_route[:k] + [customer] + new_route[k:]
                        good_route, feasible = check_potential_route(potential_route)

                        if feasible:
                            neighbor[j] = good_route
                            customer_inserted = True
                            nove_rute = neighbor
                            temp_changes.append((customer, route.list_of_customers))
                            break

                    if customer_inserted is True:
                        break

                if customer_inserted is True:
                    if temp_changes in tabu_list:
                        route_feasible = False
                    break

                route_feasible = False
                break

        if route_feasible:
            len_r = 0
            for r in neighbor:
                len_r += len(r.list_of_customers) - 2
            if len_r == num_customers:
                changes = temp_changes
                neighborhood.append(neighbor)

        b += 1

    return neighborhood, changes


def tabu_search(initial_solution, max_b):
    s = initial_solution.copy()
    s_best = initial_solution.copy()

    tabu_list = []  # list of customers
    customer_to_tabu = None

    without_route_del = 5

    del_routes = True
    num_iters = 1000
    no_change = 200
    tabu_tenure = 10

    for route in s:
        route_tup = tuple(route.list_of_customers)
        route_probs[route_tup] = 100

    for i in range(num_iters):

        if without_route_del == 0:
            del_routes = True
            without_route_del = 5

        if del_routes:
            neighborhood, changes = generate_neighborhood_del_route(s, tabu_list, max_b)
        else:
            neighborhood, changes = generate_neighborhood_changes(s, tabu_list)
            without_route_del -= 1

        shorter_neighbor_exists = False
        for j, neighbor in enumerate(neighborhood):

            if len(neighbor) < len(s):
                s = neighbor
                s_best = neighbor
                shorter_neighbor_exists = True
                s_best_string = get_solution_string(s_best, i)
                result_file.write(s_best_string)
                no_change = 200
                break

        if not shorter_neighbor_exists:
            smallest_distance = calculate_total_distance(s)
            brojac = 0
            best_neighbor = None
            for k, neighbor in enumerate(neighborhood):

                neighbor_distance = calculate_total_distance(neighbor)

                if neighbor_distance < smallest_distance:
                    best_neighbor = neighbor.copy()
                    smallest_distance = neighbor_distance
                    brojac += 1
                    customer_to_tabu = changes[k]

            if brojac == 0:

                if del_routes:
                    del_routes = False

            if best_neighbor is not None:
                s = best_neighbor.copy()

        if calculate_total_distance(s) < calculate_total_distance(s_best):
            s_best = s.copy()
            s_best_string = get_solution_string(s_best, i)
            result_file.write(s_best_string)
            no_change = 200
        else:
            no_change -= 1

        if customer_to_tabu is None:
            continue
        if len(tabu_list) < tabu_tenure:
            tabu_list.append(customer_to_tabu)
        else:
            del tabu_list[0]
            tabu_list.append(customer_to_tabu)

        for route in s_best:
            route_tup = tuple(route.list_of_customers)
            if route_tup not in route_probs:
                route_probs[route_tup] = 100

        if no_change == 0:
            break

    return s_best, calculate_total_distance(s_best)


if __name__ == '__main__':
    N = 5
    b = 10

    num_vehicles, capacity, depo_x, depo_y, depo_due_time, customers_data, depo_data, num_customers = load_data(
        instance_path)

    depo = Depo(*depo_data)

    customers = []
    for customer in customers_data:
        customers.append(Customer(*customer))

    routes = greedy_algorithm(num_vehicles, capacity, customers_data, depo_data, N)

    routes_string = get_solution_string(routes, -1)
    print("Initial solution: ")
    print(routes_string)

    print()
    best_solution, min_distance = tabu_search(routes, b)
    best_solution_string = get_solution_string(best_solution, -1)
    print("Best solution: ")
    print(best_solution_string)
