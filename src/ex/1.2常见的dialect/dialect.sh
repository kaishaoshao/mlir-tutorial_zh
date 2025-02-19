mlir_opt=../../install/bin/mlir-opt

echo "convert elementwise to linalg"
$mlir_opt -convert-elementwise-to-linalg  \
          ./add_input.mlir  -o ./add_output.mlir            

$mlir_opt -convert-elementwise-to-linalg  \
          ./add_input2.mlir  -o ./add_output2.mlir 

echo "func bufferize"
$mlir_opt -func-bufferize  \
          ./func_input.mlir  -o ./func_output.mlir   

echo "convert linalg to affine loops"
$mlir_opt -convert-elementwise-to-linalg -func-bufferize  \
          ./func_input.mlir  -o ./func_output2.mlir   

$mlir_opt  -convert-linalg-to-affine-loops \
          ./func_output2.mlir  -o ./func_output3.mlir   
    
$mlir_opt -convert-linalg-to-affine-loops -affine-loop-normalize -affine-simplify-structures  \
          ./func_output3.mlir   

    
$mlir_opt   -convert-elementwise-to-linalg \
            -func-bufferize \
            -linalg-bufferize \ #LLVM20 废弃
            -convert-linalg-to-affine-loops
            ./func64.mlir   


