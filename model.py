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
        self.stores = []
        self.regional_distribution_centers = []
        self.capacity = {}        # plant -> capacity
        self.demand = {}          # store -> demand
        self.costs = {}           # (plant, store) -> cost/unit
        self.ship = {}            # Decision variables
        self.locations = {}       # node -> {'lat': float, 'lon': float}
        #fixed variable for cost for plant ops
        #variable cost for plant ops
        
    
    def clear_model(self):
        """Clear the model."""
        self.plants = []
        self.stores = []
        self.regional_distribution_centers = []
        self.capacity = {}
        self.demand = {}
        self.costs = {}
        self.ship = {}
        self.locations = {}
        
    def add_plant(self, name, capacity, lat=None, lon=None, **kwargs):
        """
        Add a plant with its capacity and optional location.
        
        Args:
            name (str): Name of the plant
            capacity (float): Production capacity of the plant
            lat (float, optional): Latitude of the plant location
            lon (float, optional): Longitude of the plant location
            **kwargs: For backward compatibility
            
        Raises:
            ValueError: If a plant with the same name already exists
        """
        if name in self.plants:
            raise ValueError(f"A plant with name '{name}' already exists. Please use a unique name or remove the existing plant first.")
            
        self.plants.append(name)
        self.capacity[name] = capacity
        self.costs[name] = {}  # Don't initialize any costs here
        if lat is not None and lon is not None:
            self.set_location(name, lat, lon)
    
    def remove_plant(self, name):
        """Remove a plant."""
        self.plants.remove(name)
        del self.capacity[name]
        del self.costs[name]
        if name in self.locations:
            del self.locations[name]
        
    def remove_store(self, name):
        """Remove a store."""
        self.stores.remove(name)
        del self.demand[name]
        for plant in self.plants:
            del self.costs[plant][name]
        for rdc in self.regional_distribution_centers:
            if name in self.costs[rdc]:
                del self.costs[rdc][name]
        if name in self.locations:
            del self.locations[name]
        
    def remove_regional_distribution_center(self, name):
        """Remove a regional distribution center (RDC)."""
        self.regional_distribution_centers.remove(name)
        # Remove all costs to and from this RDC
        for plant in self.plants:
            if name in self.costs[plant]:
                del self.costs[plant][name]
        if name in self.costs:
            del self.costs[name]
        for s in self.stores:
            if name in self.costs and s in self.costs[name]:
                del self.costs[name][s]
        if name in self.locations:
            del self.locations[name]
        
    def add_store(self, name, demand, lat=None, lon=None, **kwargs):
        """
        Add a store with its demand and optional location.
        
        Args:
            name (str): Name of the store
            demand (float): Demand of the store
            lat (float, optional): Latitude of the store location
            lon (float, optional): Longitude of the store location
            **kwargs: For backward compatibility
            
        Raises:
            ValueError: If a store with the same name already exists
        """
        if name in self.stores:
            raise ValueError(f"A store with name '{name}' already exists. Please use a unique name or remove the existing store first.")
            
        self.stores.append(name)
        self.demand[name] = demand
        if lat is not None and lon is not None:
            self.set_location(name, lat, lon)
        # Don't initialize any costs here
    def add_regional_distribution_center(self, name, lat=None, lon=None, **kwargs):
        """
        Add a regional distribution center (RDC).
        
        Args:
            name (str): Name of the RDC
            lat (float, optional): Latitude of the RDC location
            lon (float, optional): Longitude of the RDC location
            **kwargs: For backward compatibility
            
        Raises:
            ValueError: If an RDC with the same name already exists
        """
        if name in self.regional_distribution_centers:
            raise ValueError(f"An RDC with name '{name}' already exists. Please use a unique name or remove the existing RDC first.")
            
        self.regional_distribution_centers.append(name)
        if lat is not None and lon is not None:
            self.set_location(name, lat, lon)
        # Don't initialize any costs here
    
    def set_location(self, node_name, lat, lon):
        """
        Set the location (latitude and longitude) for a node.
        
        Args:
            node_name (str): Name of the node (plant or store)
            lat (float): Latitude of the node
            lon (float): Longitude of the node
        """
        self.locations[node_name] = {'lat': lat, 'lon': lon}
    
    def get_location(self, node_name):
        """
        Get the location (latitude and longitude) of a node.
        
        Args:
            node_name (str): Name of the node (plant or store)
            
        Returns:
        dict: Dictionary with 'lat' and 'lon' keys, or None if not found
        """
        return self.locations.get(node_name)
    def get_shipping_cost(self, src, dst):
        """
        Get the shipping cost from source to destination.
        
        Args:
            src (str): Source node (plant or RDC)
            dst (str): Destination node (store or RDC)
            
        Returns:
            float: Shipping cost from source to destination, or None if not found
        """
        return self.costs.get(src, {}).get(dst)
        
    def set_shipping_cost(self, src, dst, cost):
        """Set the shipping cost from source to destination."""
        if src not in self.plants and src not in self.regional_distribution_centers:
            raise ValueError(f"Unknown source: {src}")
        if dst not in self.stores and dst not in self.regional_distribution_centers:
            raise ValueError(f"Unknown destination: {dst}")
        if src not in self.costs:
            self.costs[src] = {}
        self.costs[src][dst] = cost
            
    def update_capacity(self, plant, new_capacity):
        """Update the capacity of a plant."""
        if plant not in self.plants:
            raise ValueError(f"Unknown plant: {plant}")
        self.capacity[plant] = new_capacity
        
    def update_demand(self, s, new_demand):
        """Update the demand of a store."""
        if s not in self.stores:
            raise ValueError(f"Unknown store: {s}")
        self.demand[s] = new_demand
        
    def build_model(self):
        """Build the optimization model with the current data."""
        self.prob = LpProblem(self.name, LpMinimize)
        self.ship = {}

        # Only create variables for routes that have defined costs
        for src in self.costs:
            for dst, cost in self.costs[src].items():
                if cost > 0:  # Only create variable if cost is positive
                    self.ship[(src, dst)] = LpVariable(f"ship_{src}_{dst}", 0, None, LpInteger)

        # Objective function: minimize total cost
        self.prob += lpSum(
            self.ship[(src, dst)] * self.costs[src][dst]
            for (src, dst) in self.ship
        ), "Total_Cost"

        # Plant supply constraints
        for plant in self.plants:
            valid_rdcs = [rdc for rdc in self.regional_distribution_centers 
                        if (plant, rdc) in self.ship]
            if valid_rdcs:
                self.prob += (
                    lpSum(self.ship[(plant, rdc)] for rdc in valid_rdcs)
                    <= self.capacity[plant],
                    f"Supply_{plant}"
                )

        # RDC conservation of flow constraints
        for rdc in self.regional_distribution_centers:
            incoming = [p for p in self.plants if (p, rdc) in self.ship]
            outgoing = [s for s in self.stores if (rdc, s) in self.ship]
            if incoming or outgoing:
                self.prob += (
                    lpSum(self.ship[(p, rdc)] for p in incoming)
                    == lpSum(self.ship[(rdc, s)] for s in outgoing),
                    f"FlowConservation_{rdc}"
                )

        # Store demand constraints
        for store in self.stores:
            incoming_rdcs = [rdc for rdc in self.regional_distribution_centers 
                            if (rdc, store) in self.ship]
            if incoming_rdcs:
                self.prob += (
                    lpSum(self.ship[(rdc, store)] for rdc in incoming_rdcs)
                    >= self.demand[store],
                    f"Demand_{store}"
                )
    
    def solve(self):
        """Solve the optimization problem."""
        self.build_model()
        status = self.prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if status == LpStatusInfeasible or status == LpStatusUndefined:
            return {'status': LpStatus[status], 'total_cost': float('inf'), 'solution': {}}
    
        solution = {
            (src, dst): var.varValue
            for (src, dst), var in self.ship.items()
            if var.varValue and var.varValue > 0
        }
    
        return {
            'status': LpStatus[status],
            'total_cost': value(self.prob.objective),
            'solution': solution
        }
    
    # API is already able to understand JSON from solve() -- keeping this only for debugging
    def get_solution_summary(self):
        """Return a formatted summary of the solution."""
        result = self.solve()
        if result['status'] != 'Optimal':
            return f"Solution status: {result['status']}"
            
        output = [f"Optimal Solution Found - Total Cost: ${result['total_cost']:,.2f}\n"]
        output.append("\nShipping Plan:")
        for (src, dst), qty in result['solution'].items():
            output.append(f"Ship {int(qty)} units from {src} to {dst}")
        
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
        new_model.stores = self.stores.copy()
        new_model.regional_distribution_centers = self.regional_distribution_centers.copy()
        new_model.capacity = self.capacity.copy()
        new_model.demand = self.demand.copy()
        new_model.locations = self.locations.copy()
        
        # Deep copy the costs dictionary
        new_model.costs = {}
        for src in self.plants + self.regional_distribution_centers:
            new_model.costs[src] = self.costs[src].copy()
        
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
            'stores': {
                'added': list(set(model2.stores) - set(model1.stores)),
                'removed': list(set(model1.stores) - set(model2.stores)),
                'demand_changes': {}
            },
            'regional_distribution_centers': {
                'added': list(set(model2.regional_distribution_centers) - set(model1.regional_distribution_centers)),
                'removed': list(set(model1.regional_distribution_centers) - set(model2.regional_distribution_centers)),
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
        
        # Check for demand changes in common stores
        common_stores = set(model1.stores) & set(model2.stores)
        for s in common_stores:
            if model1.demand.get(s) != model2.demand.get(s):
                differences['stores']['demand_changes'][s] = (
                    model1.demand.get(s),
                    model2.demand.get(s)
                )
        
        # Check for cost changes
        for src in common_plants | set(model1.regional_distribution_centers) & set(model2.regional_distribution_centers):
            for dst in common_stores | set(model1.regional_distribution_centers) & set(model2.regional_distribution_centers):
                cost1 = model1.costs.get(src, {}).get(dst)
                cost2 = model2.costs.get(src, {}).get(dst)
                if cost1 != cost2:
                    if src not in differences['cost_changes']:
                        differences['cost_changes'][src] = {}
                    differences['cost_changes'][src][dst] = (cost1, cost2)
        
        return differences

# Example usage
if __name__ == "__main__":
    # Create model instance
    model = NetworkOptimization("US_Transportation_Network")
    
    # Add plants with capacities (in units) and locations
    model.add_plant("Chicago_IL", 13000, 41.8781, -87.6298)    # Major manufacturing hub
    model.add_plant("Houston_TX", 14500, 29.7604, -95.3698)     # Major industrial center
    model.add_plant("LosAngeles_CA", 12500, 34.0522, -118.2437)  # West Coast manufacturing
    
    # Add regional distribution centers
    model.add_regional_distribution_center("RDC1_Chicago", 41.8781, -87.6298)  # Midwest hub
    model.add_regional_distribution_center("RDC2_Atlanta", 33.7490, -84.3880)  # Southeast hub
    #model.add_regional_distribution_center("RDC3_Denver", 39.7392, -104.9903)  # Central hub
    
    # Add stores with demands (in units) and locations
    model.add_store("NewYork_NY", 7000, 40.7128, -74.0060)     # High population density
    model.add_store("Miami_FL", 5500, 25.7617, -80.1918)       # Southeast retail hub
    model.add_store("Seattle_WA", 5000, 47.6062, -122.3321)    # Pacific Northwest
    model.add_store("Dallas_TX", 6500, 32.7767, -96.7970)      # Major Texas market
    model.add_store("Boston_MA", 5000, 42.3601, -71.0589)      # Major Northeast market
    model.add_store("LasVegas_NV", 8000, 36.1699, -115.1398)      # Major West Coast market
    
    # Set shipping costs (in dollars per unit)
    # Plants to RDCs
    model.set_shipping_cost("Chicago_IL", "RDC1_Chicago", 100)
    model.set_shipping_cost("Chicago_IL", "RDC2_Atlanta", 100)
    model.set_shipping_cost("Houston_TX", "RDC2_Atlanta", 800)
    model.set_shipping_cost("Houston_TX", "RDC1_Chicago", 1000)
    model.set_shipping_cost("LosAngeles_CA", "RDC2_Atlanta", 3000)
    model.set_shipping_cost("LosAngeles_CA", "RDC1_Chicago", 3500)
    
    # RDCs to Stores
    model.set_shipping_cost("RDC1_Chicago", "NewYork_NY", 800)
    model.set_shipping_cost("RDC2_Atlanta", "Miami_FL", 600)
    model.set_shipping_cost("RDC2_Atlanta", "Dallas_TX", 500)
    model.set_shipping_cost("RDC1_Chicago", "Seattle_WA", 1000)
    model.set_shipping_cost("RDC2_Atlanta", "Dallas_TX", 800)
    model.set_shipping_cost("RDC1_Chicago", "LasVegas_NV", 1200)
    model.set_shipping_cost("RDC2_Atlanta", "LasVegas_NV", 1200)
    model.set_shipping_cost("RDC1_Chicago", "Boston_MA", 800)
    model.set_shipping_cost("RDC2_Atlanta", "Boston_MA", 1000)
    model.set_shipping_cost("RDC2_Atlanta","Seattle_WA", 1000)
   
    
    
    # Solve and print results
    solution = model.solve()
    print("\nOptimal Solution:")
    print(model.get_solution_summary())

    # Check if solution has 'solution' key and it's not None
    if 'solution' in solution and solution['solution']:
        for (src, dst), amount in solution['solution'].items():
            if amount > 0:  # Only show routes with shipments
                print(f"Ship {int(amount)} units from {src} to {dst}")

    print("\nTotal Cost: $", solution.get('total_cost', 'N/A'))