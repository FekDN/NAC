import math
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Generate three separate files for maximum accuracy
OUTPUT_FILENAME_GELU = "nac_gelu_lut.mem"
OUTPUT_FILENAME_TANH = "nac_tanh_lut.mem"
OUTPUT_FILENAME_ERF  = "nac_erf_lut.mem" # New file for ERF

LUT_SIZE = 2048
FIXED_POINT_Q = 16  # Q16.16 format
LSB_BIT_INDEX = 9   # Corresponds to Verilog op_a[19:9], giving a range of +/- 8.0

# ==============================================================================
# MATH HELPERS
# ==============================================================================
def float_to_q16_16(val):
    """Converts python float to 32-bit hex string (Q16.16 two's complement)."""
    scaled = int(val * (2**FIXED_POINT_Q))
    
    max_val = (2**31) - 1
    min_val = -(2**31)
    scaled = max(min(scaled, max_val), min_val)
    
    if scaled < 0:
        scaled = (1 << 32) + scaled
        
    return f"{scaled:08X}"

def gelu(x):
    """Gaussian Error Linear Unit approximation."""
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * math.pow(x, 3))))

def tanh(x):
    """Hyperbolic tangent."""
    return math.tanh(x)

def erf(x):
    """Error function. Python 3 has this built-in."""
    return math.erf(x)

# ==============================================================================
# MAIN GENERATION
# ==============================================================================
def generate_lut(filename: str, func):
    """Generic function to generate a LUT file."""
    print(f"Generating {filename} for {func.__name__.upper()}...")
    
    step_exponent = LSB_BIT_INDEX - FIXED_POINT_Q
    step_size = 2 ** step_exponent
    
    min_input = -1024 * step_size
    max_input = 1023 * step_size
    print(f"  - LUT Input Range: [{min_input:.4f} to {max_input:.4f}]")
    print(f"  - LUT Step Size: {step_size:.6f}")

    try:
        with open(filename, 'w') as f:
            for i in range(LUT_SIZE):
                offset = i - 1024
                x_val = offset * step_size
                y_val = func(x_val)
                hex_val = float_to_q16_16(y_val)
                f.write(f"{hex_val}\n")
        print(f"Done. File saved to {filename}\n")
    except IOError as e:
        print(f"Error writing to file {filename}: {e}\n")


def main():
    print(f"Generating Activation LUTs in Q16.16 format...")
    print(f"LUT Size: {LUT_SIZE}, Verilog Slice LSB: {LSB_BIT_INDEX}\n")
    
    # Generate a LUT for each function
    generate_lut(OUTPUT_FILENAME_GELU, gelu)
    generate_lut(OUTPUT_FILENAME_TANH, tanh)
    generate_lut(OUTPUT_FILENAME_ERF, erf)
    
if __name__ == "__main__":
    main()