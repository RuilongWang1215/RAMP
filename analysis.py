import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import CityGraphModel, ModelConfig

# --- Helper to generate consistent city data ---
def get_synthetic_city(seed=42):
    np.random.seed(seed)
    df = pd.DataFrame({
        'id': range(50),
        'x': np.random.uniform(0, 10, 50),
        'y': np.random.uniform(0, 10, 50) 
    })
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))

# ==========================================
# TEST 1: REPRODUCIBILITY CHECK
# ==========================================
def check_reproducibility():
    print("\n--- Test 1: Reproducibility Check ---")
    stops = get_synthetic_city()
    
    # Run A
    cfg = ModelConfig()
    model_a = CityGraphModel(stops, cfg=cfg, seed=999)
    model_a.run()
    res_a = model_a.datacollector.get_model_vars_dataframe().iloc[-1]['total_water_removed']
    
    # Run B (Same Seed)
    model_b = CityGraphModel(stops, cfg=cfg, seed=999)
    model_b.run()
    res_b = model_b.datacollector.get_model_vars_dataframe().iloc[-1]['total_water_removed']
    
    # Run C (Different Seed)
    model_c = CityGraphModel(stops, cfg=cfg, seed=123)
    model_c.run()
    res_c = model_c.datacollector.get_model_vars_dataframe().iloc[-1]['total_water_removed']
    
    if res_a == res_b:
        print(f"✅ PASSED: Run A ({res_a}) == Run B ({res_b}) with same seed.")
    else:
        print(f"❌ FAILED: Run A ({res_a}) != Run B ({res_b}) with same seed.")
        
    if res_a != res_c:
        print(f"✅ PASSED: Run A ({res_a}) != Run C ({res_c}) with different seed.")
    else:
        print(f"⚠️ WARNING: Run A and Run C produced identical results. Check randomness.")

# ==========================================
# TEST 2: BOUNDARY / STRESS TESTS
# ==========================================
def check_boundaries():
    print("\n--- Test 2: Boundary Stress Tests ---")
    stops = get_synthetic_city()
    
    # Case 1: Zero Crews
    print("Running with 0 Crews...")
    cfg_zero = ModelConfig(num_pump_crews=0)
    model_zero = CityGraphModel(stops, cfg=cfg_zero, seed=42)
    model_zero.run()
    water_zero = model_zero.total_water_removed
    
    # Case 2: Massive Fleet
    print("Running with 50 Crews...")
    cfg_max = ModelConfig(num_pump_crews=50)
    model_max = CityGraphModel(stops, cfg=cfg_max, seed=42)
    model_max.run()
    water_max = model_max.total_water_removed
    
    # Assertions
    if water_zero == 0:
        print(f"✅ PASSED: 0 Crews removed 0 liters.")
    else:
        print(f"❌ FAILED: 0 Crews removed {water_zero} liters (Magic water removal?).")
        
    if water_max > 100000: # Arbitrary high number
        print(f"✅ PASSED: 50 Crews removed significant water ({water_max:,.0f} L).")
    else:
        print(f"❌ FAILED: 50 Crews barely removed water. Logic bottleneck suspected.")

# ==========================================
# TEST 3: SENSITIVITY ANALYSIS
# ==========================================
def run_sensitivity_analysis():
    print("\n--- Test 3: Sensitivity Analysis (Fleet Size vs Speed) ---")
    stops = get_synthetic_city()
    
    results = []
    
    # Parameter Sweep
    fleet_sizes = [1, 3, 5, 10, 15]
    speeds = [0.5, 1.0, 2.0] # Slow, Medium, Fast
    
    total_runs = len(fleet_sizes) * len(speeds)
    count = 0
    
    for n in fleet_sizes:
        for s in speeds:
            count += 1
            print(f"   Running scenario {count}/{total_runs}: {n} crews at speed {s}...")
            
            cfg = ModelConfig(
                num_pump_crews=n, 
                crew_speed=s,
                dispatch_strategy='NEAREST'
            )
            # Use fixed seed for fair comparison
            model = CityGraphModel(stops, cfg=cfg, seed=42)
            model.run()
            
            final_data = model.datacollector.get_model_vars_dataframe().iloc[-1]
            
            results.append({
                'Num_Crews': n,
                'Speed': s,
                'Total_Removed': final_data['total_water_removed'],
                'Max_Flood_Depth': final_data['max_mean_flood']
            })
            
    df_res = pd.DataFrame(results)
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Water Removed
    plt.subplot(1, 2, 1)
    sns.lineplot(data=df_res, x='Num_Crews', y='Total_Removed', hue='Speed', marker='o', palette='viridis')
    plt.title("Sensitivity: Water Removed")
    plt.ylabel("Liters Removed")
    plt.grid(True)
    
    # Plot 2: Flood Mitigation
    plt.subplot(1, 2, 2)
    sns.lineplot(data=df_res, x='Num_Crews', y='Max_Flood_Depth', hue='Speed', marker='o', palette='magma')
    plt.title("Sensitivity: Flood Control")
    plt.ylabel("Max Mean Flood Depth (m)")
    plt.grid(True)
    
    plt.tight_layout()
    print("Done. Displaying plots...")
    plt.show()

if __name__ == "__main__":
    check_reproducibility()
    check_boundaries()
    run_sensitivity_analysis()