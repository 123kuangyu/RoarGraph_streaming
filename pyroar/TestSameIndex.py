#!/usr/bin/env python3
import os
import struct
import sys

def load_and_compare_projection_graphs(filename1, filename2):
    """
    Compare two binary index files to check if they have identical content.
    Each file follows the format written by SaveProjectionGraph:
    - projection_ep_ (uint32)
    - u32_nd_ (uint32)
    - For each node i (0 to u32_nd_-1):
        - nbr_size (uint32): number of neighbors
        - neighbors (array of uint32): list of neighbor IDs
    """
    try:
        with open(filename1, 'rb') as file1, open(filename2, 'rb') as file2:
            # Read header data
            projection_ep1 = struct.unpack('<I', file1.read(4))[0]
            u32_nd1 = struct.unpack('<I', file1.read(4))[0]
            
            projection_ep2 = struct.unpack('<I', file2.read(4))[0]
            u32_nd2 = struct.unpack('<I', file2.read(4))[0]
            
            # Compare header data
            if projection_ep1 != projection_ep2 or u32_nd1 != u32_nd2:
                print("Header information differs:")
                print(f"File 1: projection_ep={projection_ep1}, u32_nd={u32_nd1}")
                print(f"File 2: projection_ep={projection_ep2}, u32_nd={u32_nd2}")
                return False
            
            print(f"Header info: projection_ep={projection_ep1}, u32_nd={u32_nd1}")
            
            # Compare graph data for each node
            for i in range(u32_nd1):
                # Read neighbor size for both files
                nbr_size1 = struct.unpack('<I', file1.read(4))[0]
                nbr_size2 = struct.unpack('<I', file2.read(4))[0]
                
                if nbr_size1 != nbr_size2:
                    print(f"Difference found at node {i}:")
                    print(f"File 1 has {nbr_size1} neighbors")
                    print(f"File 2 has {nbr_size2} neighbors")
                    return False
                
                # Read and compare neighbor data
                neighbors1_bytes = file1.read(4 * nbr_size1)
                neighbors2_bytes = file2.read(4 * nbr_size2)
                
                if neighbors1_bytes != neighbors2_bytes:
                    # Detailed comparison to find exact mismatch
                    neighbors1 = struct.unpack(f'<{nbr_size1}I', neighbors1_bytes)
                    neighbors2 = struct.unpack(f'<{nbr_size2}I', neighbors2_bytes)
                    
                    for j in range(nbr_size1):
                        if neighbors1[j] != neighbors2[j]:
                            print(f"Difference found at node {i}, neighbor {j}:")
                            print(f"File 1: {neighbors1[j]}")
                            print(f"File 2: {neighbors2[j]}")
                            return False
                
                # Progress indicator for large files
                if i > 0 and i % 100000 == 0:
                    print(f"Compared {i} nodes out of {u32_nd1}")
            
            # Check if we've reached the end of both files
            remaining1 = file1.read(1)
            remaining2 = file2.read(1)
            
            if remaining1 or remaining2:
                print("One file has additional data after expected end")
                return False
            
            return True
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def main():
    
    ProjectPath= os.path.join(os.path.dirname(os.path.dirname(__file__)))
    data_path= os.path.join(ProjectPath, "data", "t2i-10M")
    file1 = os.path.join(data_path, "t2i_10M_roar10_0.index")
    file2 = os.path.join(data_path,"t2i_10M_roar10.index")
    print("Comparing index files:")
    print(f"  File 1: {file1}")
    print(f"  File 2: {file2}")
    
    for filepath in [file1, file2]:
        if not os.path.exists(filepath):
            print(f"Error: File does not exist: {filepath}")
            return 2
    
    file1_size = os.path.getsize(file1)
    file2_size = os.path.getsize(file2)
    
    print(f"File 1 size: {file1_size} bytes")
    print(f"File 2 size: {file2_size} bytes")
    
    # if file1_size != file2_size:
    #     print("Files have different sizes - they cannot be identical")
    #     return 1
    
    identical = load_and_compare_projection_graphs(file1, file2)
    
    if identical:
        print("The index files are identical.")
        return 0
    else:
        print("The index files differ.")
        return 1

if __name__ == "__main__":
    sys.exit(main())