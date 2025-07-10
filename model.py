from pulp import *

class NetworkOptimization:
    def __init__(self, name="Network_Optimization_Model"):
        """
        Initialize the network optimization model.
        
        Args:
            name (str): Name of the optimization problem
        """
        self.name = name
        self.prob = LpProblem(name, LpMinimize)
        
        # Initialize data structures
        self.plants = []
        self.distribution_centers = []
        self.capacity = {}        # plant -> capacity
        self.demand = {}          # dc -> demand
        self.costs = {}           # (plant, dc) -> cost/unit
        self.ship = {}            # Decision variables
        #fixed variable for cost for plant ops
        #variable cost for plant ops
        
    def add_plant(self, name, capacity):
        """Add a plant with its capacity."""
        self.plants.append(name)
        self.capacity[name] = capacity
        self.costs[name] = {}
    def remove_plant(self, name):
        """Remove a plant."""
        self.plants.remove(name)
        del self.capacity[name]
        del self.costs[name]
    def remove_distribution_center(self, name):
        """Remove a distribution center."""
        self.distribution_centers.remove(name)
        del self.demand[name]
        for plant in self.plants:
            del self.costs[plant][name]
    def add_distribution_center(self, name, demand):
        """Add a distribution center with its demand."""
        self.distribution_centers.append(name)
        self.demand[name] = demand
        for plant in self.plants:
            self.costs[plant][name] = 0  # Initialize cost to 0
            
    def set_shipping_cost(self, plant, dc, cost):
        """Set the shipping cost from plant to distribution center."""
        if plant not in self.plants:
            raise ValueError(f"Unknown plant: {plant}")
        if dc not in self.distribution_centers:
            raise ValueError(f"Unknown distribution center: {dc}")
        self.costs[plant][dc] = cost
        
    def update_capacity(self, plant, new_capacity):
        """Update the capacity of a plant."""
        if plant not in self.plants:
            raise ValueError(f"Unknown plant: {plant}")
        self.capacity[plant] = new_capacity
        
    def update_demand(self, dc, new_demand):
        """Update the demand of a distribution center."""
        if dc not in self.distribution_centers:
            raise ValueError(f"Unknown distribution center: {dc}")
        self.demand[dc] = new_demand
        
    def build_model(self):
        """Build the optimization model with the current data."""
        # Clear any existing variables
        self.prob = LpProblem(self.name, LpMinimize)
        self.ship = {}
        
        # Create decision variables
        for p in self.plants:
            for d in self.distribution_centers:
                self.ship[(p, d)] = LpVariable(f"ship_{p}_{d}", 0, None, LpInteger)
        
        # Objective function: minimize total cost
        self.prob += lpSum(self.ship[(p, d)] * self.costs[p][d] 
                          for p in self.plants 
                          for d in self.distribution_centers), "Total_Cost"
        
        # Supply constraints
        for p in self.plants:
            self.prob += (lpSum(self.ship[(p, d)] for d in self.distribution_centers) 
                         <= self.capacity[p], f"Supply_{p}")
            
        # Demand constraints
        for d in self.distribution_centers:
            self.prob += (lpSum(self.ship[(p, d)] for p in self.plants) 
                         >= self.demand[d], f"Demand_{d}")
    
    def solve(self):
        """Solve the optimization problem."""
        self.build_model()
        status = self.prob.solve(pulp.PULP_CBC_CMD(msg=False))
        return {
            'status': LpStatus[status],
            'total_cost': value(self.prob.objective),
            'solution': {
                (p, d): self.ship[(p, d)].varValue
                for p in self.plants
                for d in self.distribution_centers
                if self.ship[(p, d)].varValue > 0
            }
        }
    
    # API is already able to understand JSON from solve() -- keeping this only for debugging
    def get_solution_summary(self):
        """Return a formatted summary of the solution."""
        result = self.solve()
        if result['status'] != 'Optimal':
            return f"Solution status: {result['status']}"
            
        output = [f"Optimal Solution Found - Total Cost: ${result['total_cost']:,.2f}\n"]
        output.append("\nShipping Plan:")
        for (p, d), qty in result['solution'].items():
            output.append(f"Ship {int(qty)} units from {p} to {d}")
        
        return "\n".join(output)

# Example usage
if __name__ == "__main__":
    # Create model instance
    model = NetworkOptimization("Railey_Transportation_Problem")
    
    # Add plants with capacities
    model.add_plant("Ahmedabad", 4800)
    model.add_plant("Patna", 3500)
    model.add_plant("Hyderabad", 2200)
    
    # Add distribution centers with demands
    model.add_distribution_center("Bhopal", 4800)
    model.add_distribution_center("Indore", 5700)
    
    # Set shipping costs
    model.set_shipping_cost("Ahmedabad", "Bhopal", 16500)
    model.set_shipping_cost("Ahmedabad", "Indore", 10600)
    model.set_shipping_cost("Patna", "Bhopal", 12200)
    model.set_shipping_cost("Patna", "Indore", 12600)
    model.set_shipping_cost("Hyderabad", "Bhopal", 10300)
    model.set_shipping_cost("Hyderabad", "Indore", 9240)
    
    # Solve and print results
    print(model.get_solution_summary())