import math

# ==============================================================================
# CONFIGURATION
# ==============================================================================
OUTPUT_FILENAME = "nac_activation_lut.mem"
LUT_SIZE = 2048
FIXED_POINT_Q = 16  # Q16.16 format

# ------------------------------------------------------------------------------
# HARDWARE ADDRESSING CONFIG
# ------------------------------------------------------------------------------
# In the NAC_ALU_Core.v file the line: lut_idx = op_a[16:6] + 11'd1024;
# This means that the LSB of the address corresponds to bit 6 of the input number.
# Bit 6 in Q16.16 weighs 2^(-10) = 1/1024.
#
# IMPORTANT: When LSB_BIT_INDEX = 6, the range of LUT input values ​​will be from -1.0 to +1.0.
# For GELU this is too low (it saturates/linearizes at +/- 3.0...4.0).
#
# RECOMMENDATION: Change the line in Verilog to: lut_idx = op_a[19:9] + 11'd1024;
# and set LSB_BIT_INDEX = 9 here. This will give a range of +/- 8.0.
#
# For now, generate code strictly for the current Verilog:
LSB_BIT_INDEX = 9  
# ------------------------------------------------------------------------------

# Select function: 'gelu' or 'tanh'
ACTIVE_FUNCTION = 'gelu' 

# ==============================================================================
# MATH HELPERS
# ==============================================================================
def float_to_q16_16(val):
    """Converts python float to 32-bit hex string (Q16.16 two's complement)."""
    # Scale
    scaled = int(val * (2**FIXED_POINT_Q))
    
    # Clamp to 32-bit signed range
    max_val = (2**31) - 1
    min_val = -(2**31)
    if scaled > max_val: scaled = max_val
    if scaled < min_val: scaled = min_val
    
    # Handle negative numbers (Two's complement)
    if scaled < 0:
        scaled = (1 << 32) + scaled
        
    return f"{scaled:08X}"

def gelu(x):
    """Gaussian Error Linear Unit approximation."""
    # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * math.pow(x, 3))))

def tanh(x):
    return math.tanh(x)

# ==============================================================================
# MAIN GENERATION
# ==============================================================================
def main():
    print(f"Generating {OUTPUT_FILENAME}...")
    print(f"Function: {ACTIVE_FUNCTION.upper()}")
    print(f"Format: Q16.16")
    
    # Calculate step size based on hardware LSB extraction
    # Q16.16 means bit 0 is 2^-16.
    # If hardware uses bit N as LSB, the step is 2^(N - 16).
    step_exponent = LSB_BIT_INDEX - FIXED_POINT_Q
    step_size = 2 ** step_exponent
    
    print(f"Verilog Slice LSB: {LSB_BIT_INDEX}")
    print(f"LUT Step Size: {step_size:.6f}")
    
    # Calculate covered range
    min_input = -1024 * step_size
    max_input = 1023 * step_size
    print(f"LUT Input Range: [{min_input:.4f} to {max_input:.4f}]")
    
    if abs(max_input) < 2.0 and ACTIVE_FUNCTION == 'gelu':
        print("\n[WARNING] The range is very narrow for GELU. Consider changing Verilog to use op_a[19:9].")

    with open(OUTPUT_FILENAME, 'w') as f:
        # LUT addresses 0 to 2047
        # Hardware logic: index 1024 corresponds to input 0.
        # i = 0    -> input = -1024 * step
        # i = 1024 -> input = 0
        # i = 2047 -> input = 1023 * step
        
        for i in range(LUT_SIZE):
            # Calculate signed offset from center
            offset = i - 1024
            
            # Calculate real input value X
            x_val = offset * step_size
            
            # Calculate Y
            if ACTIVE_FUNCTION == 'tanh':
                y_val = tanh(x_val)
            else:
                y_val = gelu(x_val)
            
            # Convert to Hex
            hex_val = float_to_q16_16(y_val)
            
            # Write to file
            f.write(f"{hex_val}\n")
            
    print(f"Done. File saved to {OUTPUT_FILENAME}")

if __name__ == "__main__":
    main()