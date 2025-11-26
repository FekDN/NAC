`include "nac_defines.vh"

module NAC_Byte_Fetcher #(
    parameter FIFO_DEPTH_LOG2 = 2 // 2^2 = 4 words (16 bytes) buffer depth
)(
    input  wire        clk,
    input  wire        rst_n,

    // ========================================================================
    // Control Interface
    // ========================================================================
    input  wire        flush,        // Active High: Clears buffer, resets address
    input  wire [31:0] start_addr,   // Byte-aligned start address (e.g. 0x1003)
    output wire        busy,         // High if buffer is empty (Starvation)

    // ========================================================================
    // Consumer Interface (To NAC Decoder)
    // ========================================================================
    input  wire        byte_req,     // Decoder requests a byte
    output reg  [7:0]  byte_data,    // The byte payload
    output reg         byte_valid,   // Data is valid this cycle

    // ========================================================================
    // Memory Provider Interface (To Arbiter/DDR)
    // ========================================================================
    output reg  [31:0] mem_addr,
    output reg         mem_req,      // Read Strobe
    input  wire        mem_grant,    // Arbiter accepted the request
    input  wire        mem_valid,    // Data returning from DDR
    input  wire [31:0] mem_rdata     // 32-bit Data
);

    localparam FIFO_DEPTH = (1 << FIFO_DEPTH_LOG2);

    // ========================================================================
    // Internal State
    // ========================================================================
    
    // 1. Fetch Address Management
    reg [31:0] current_fetch_addr; // Aligned to 32-bit boundaries

    // 2. FIFO Storage (Ring Buffer)
    reg [31:0] fifo_data [0:FIFO_DEPTH-1];
    reg [FIFO_DEPTH_LOG2-1:0] wr_ptr;
    reg [FIFO_DEPTH_LOG2-1:0] rd_ptr;
    reg [FIFO_DEPTH_LOG2:0]   words_in_fifo; // Counter 0..4

    // 3. Byte Extraction
    reg [1:0]  byte_offset; // Current byte pointer within the 32-bit word (0..3)

    // 4. Flow Control
    reg [FIFO_DEPTH_LOG2:0] pending_req_cnt; // Number of requests sent but not returned
    
    // Starvation logic: Busy if we have no valid bytes ready for the decoder
    assign busy = (words_in_fifo == 0);

    // ========================================================================
    // Main Logic
    // ========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset State
            mem_req <= 0;
            mem_addr <= 0;
            byte_valid <= 0;
            byte_data <= 0;
            
            wr_ptr <= 0;
            rd_ptr <= 0;
            words_in_fifo <= 0;
            byte_offset <= 0;
            
            current_fetch_addr <= 0;
            pending_req_cnt <= 0;
        end 
        else if (flush) begin
            // ----------------------------------------------------------------
            // FLUSH: Handling Jumps / Calls / Returns
            // ----------------------------------------------------------------
            // Immediately invalidate FIFO and align address
            mem_req <= 0;
            byte_valid <= 0;
            
            wr_ptr <= 0;
            rd_ptr <= 0;
            words_in_fifo <= 0;
            pending_req_cnt <= 0; // Ignore in-flight data from old stream
            
            // Align address to 4-byte boundary: (addr & 0xFFFFFFFC)
            current_fetch_addr <= {start_addr[31:2], 2'b00};
            
            // Set the byte offset (0, 1, 2, or 3) based on LSBs
            byte_offset <= start_addr[1:0];
        end 
        else begin
            // ----------------------------------------------------------------
            // 1. PRODUCER: Request Logic (Prefetcher)
            // ----------------------------------------------------------------
            // Logic: Can we send a request? 
            // Condition: (Space in FIFO) - (Requests already in flight) > 0
            // AND we are not currently asserting a request that hasn't been granted.
            
            if ( !mem_req && (words_in_fifo + pending_req_cnt < FIFO_DEPTH) ) begin
                mem_addr <= current_fetch_addr;
                mem_req  <= 1;
            end

            // Handle Bus Grant (Handshake)
            if (mem_req && mem_grant) begin
                mem_req <= 0; // Deassert strobe
                pending_req_cnt <= pending_req_cnt + 1;
                current_fetch_addr <= current_fetch_addr + 4; // Advance aligned address
            end

            // ----------------------------------------------------------------
            // 2. PRODUCER: Response Logic (Data Ingestion)
            // ----------------------------------------------------------------
            // Note: If we flushed, pending_req_cnt is 0. 
            // If data comes back after flush, we must discard it (stale data).
            // We implement this by checking pending_req_cnt > 0.
            
            if (mem_valid) begin
                if (pending_req_cnt > 0) begin
                    // Valid data for current stream
                    fifo_data[wr_ptr] <= mem_rdata;
                    wr_ptr <= wr_ptr + 1;
                    words_in_fifo <= words_in_fifo + 1;
                    pending_req_cnt <= pending_req_cnt - 1;
                end 
                // else: Stale data from pre-flush request, auto-discarded
            end

            // ----------------------------------------------------------------
            // 3. CONSUMER: Byte Extraction Logic
            // ----------------------------------------------------------------
            byte_valid <= 0;

            if (byte_req && words_in_fifo > 0) begin
                byte_valid <= 1;

                // Extract Byte based on Offset (Big Endian / Network Order)
                // [31:24] -> [23:16] -> [15:8] -> [7:0]
                case (byte_offset)
                    2'b00: byte_data <= fifo_data[rd_ptr][31:24];
                    2'b01: byte_data <= fifo_data[rd_ptr][23:16];
                    2'b10: byte_data <= fifo_data[rd_ptr][15:8];
                    2'b11: byte_data <= fifo_data[rd_ptr][7:0];
                endcase

                // Pointer Management
                if (byte_offset == 3) begin
                    // Word consumed completely
                    byte_offset <= 0;
                    rd_ptr <= rd_ptr + 1;
                    words_in_fifo <= words_in_fifo - 1;
                end else begin
                    // Move to next byte in same word
                    byte_offset <= byte_offset + 1;
                end
            end
        end
    end

endmodule
