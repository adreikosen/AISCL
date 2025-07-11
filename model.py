from pulp import *
from copy import deepcopy

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
        
    
    def clear_model(self):
        """Clear the model."""
        self.plants = []
        self.distribution_centers = []
        self.capacity = {}
        self.demand = {}
        self.costs = {}
        self.ship = {}
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
        if status == LpStatusInfeasible or status == LpStatusUndefined:
            return { 'There was an error solving the problem. Please check your model.'}
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

    def copy(self, new_name=None):
        """
        Create a deep copy of the current model.
        
        Args:
            new_name (str, optional): Name for the new model. If None, appends "_copy" to the original name.
            
        Returns:
            NetworkOptimization: A new instance with the same data as the current model
        """
        if new_name is None:
            new_name = f"{self.name}_copy"
            
        # Create new instance
        new_model = NetworkOptimization(new_name)
        
        # Copy all attributes
        new_model.plants = self.plants.copy()
        new_model.distribution_centers = self.distribution_centers.copy()
        new_model.capacity = self.capacity.copy()
        new_model.demand = self.demand.copy()
        
        # Deep copy the costs dictionary
        new_model.costs = {}
        for plant in self.plants:
            new_model.costs[plant] = self.costs[plant].copy()
        
        return new_model

    @staticmethod
    def compare_models(model1, model2):
        """
        Compare two NetworkOptimization models and return the differences.
        
        Args:
            model1 (NetworkOptimization): First model to compare
            model2 (NetworkOptimization): Second model to compare
            
        Returns:
            dict: Dictionary containing the differences between the two models
        """
        differences = {
            'name': (model1.name, model2.name),
            'plants': {
                'added': list(set(model2.plants) - set(model1.plants)),
                'removed': list(set(model1.plants) - set(model2.plants)),
                'capacity_changes': {}
            },
            'distribution_centers': {
                'added': list(set(model2.distribution_centers) - set(model1.distribution_centers)),
                'removed': list(set(model1.distribution_centers) - set(model2.distribution_centers)),
                'demand_changes': {}
            },
            'cost_changes': {}
        }
        
        # Check for capacity changes in common plants
        common_plants = set(model1.plants) & set(model2.plants)
        for plant in common_plants:
            if model1.capacity.get(plant) != model2.capacity.get(plant):
                differences['plants']['capacity_changes'][plant] = (
                    model1.capacity.get(plant),
                    model2.capacity.get(plant)
                )
        
        # Check for demand changes in common distribution centers
        common_dcs = set(model1.distribution_centers) & set(model2.distribution_centers)
        for dc in common_dcs:
            if model1.demand.get(dc) != model2.demand.get(dc):
                differences['distribution_centers']['demand_changes'][dc] = (
                    model1.demand.get(dc),
                    model2.demand.get(dc)
                )
        
        # Check for cost changes
        for plant in common_plants:
            for dc in common_dcs:
                cost1 = model1.costs.get(plant, {}).get(dc)
                cost2 = model2.costs.get(plant, {}).get(dc)
                if cost1 != cost2:
                    if plant not in differences['cost_changes']:
                        differences['cost_changes'][plant] = {}
                    differences['cost_changes'][plant][dc] = (cost1, cost2)
        
        return differences

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

    # Create a copy of the model
    model_copy = model.copy()

    # Compare the two models
    differences = NetworkOptimization.compare_models(model, model_copy)
    print(differences)