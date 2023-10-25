#include "HYPRE_struct_ls.h"
#include "mpi.h"
#include <cuda_runtime.h>

int main(int argc, char *argv[])
{
    int myid, num_procs;

    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    /* Initialize HYPRE */
    HYPRE_Init();

    // HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST);
    // HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST);
    HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
    HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);

    HYPRE_StructGrid grid;
    int ndim = 2;
    int **ilower = (int **)calloc(2, sizeof(int *));
    int **iupper = (int **)calloc(2, sizeof(int *));
    ilower[0] = (int *)calloc(2, sizeof(int));
    ilower[1] = (int *)calloc(2, sizeof(int));
    iupper[0] = (int *)calloc(2, sizeof(int));
    iupper[1] = (int *)calloc(2, sizeof(int));

    // int ilower[2][2] = {{-3, 1}, {0, 1}};
    // int iupper[2][2] = {{-1, 2}, {2, 4}};

    /* Create the grid object */
    HYPRE_StructGridCreate(MPI_COMM_WORLD, ndim, &grid);

    /* Set grid extents for the first box */
    HYPRE_StructGridSetExtents(grid, ilower[0], iupper[0]);

    /* Set grid extents for the second box */
    HYPRE_StructGridSetExtents(grid, ilower[1], iupper[1]);

    /* Assemble the grid */
    HYPRE_StructGridAssemble(grid);

    HYPRE_StructStencil stencil;
    // int ndim = 2;
    int size = 5;
    int entry;
    int **offsets = (int **)calloc(5, sizeof(int *));
    for (int i = 0; i < size; i++)
        offsets[i] = (int *)calloc(2, sizeof(int));

    offsets[0][0] = 0;
    offsets[0][1] = 0;
    offsets[1][0] = -1;
    offsets[1][1] = 0;
    offsets[2][0] = 1;
    offsets[2][1] = 0;
    offsets[3][0] = 0;
    offsets[3][1] = -1;
    offsets[4][0] = 0;
    offsets[4][1] = 1;
    // int offsets[5][2] = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};

    /* Create the stencil object */
    HYPRE_StructStencilCreate(ndim, size, &stencil);

    /* Set stencil entries */
    for (entry = 0; entry < size; entry++)
    {
        HYPRE_StructStencilSetElement(stencil, entry, offsets[entry]);
    }

    /* Thats it!  There is no assemble routine */

    HYPRE_StructMatrix A;
    // double values[36];
    // double *values = (double *)calloc(36, sizeof(double));
    double *values;
    cudaMallocManaged(&values, 36 * sizeof(double));
    // int stencil_indices[2] = {0, 3};
    int *stencil_indices = (int *)calloc(2, sizeof(int));
    stencil_indices[0] = 0;
    stencil_indices[0] = 3;
    int i;

    HYPRE_StructMatrixCreate(MPI_COMM_WORLD, grid, stencil, &A);
    HYPRE_StructMatrixInitialize(A);

    for (i = 0; i < 36; i += 2)
    {
        values[i] = 4.0;
        values[i + 1] = -1.0;
    }

    // If this is commented out the programs finishes without errors.
    HYPRE_StructMatrixSetBoxValues(A, ilower[0], iupper[0], 2,
                                   stencil_indices, values);
    HYPRE_StructMatrixSetBoxValues(A, ilower[1], iupper[1], 2,
                                   stencil_indices, values);

    /* set boundary conditions */
    // not done here

    HYPRE_StructMatrixAssemble(A);

    // HYPRE_StructVector b;
    // double values_b[18];
    // // int i;

    // HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &b);
    // HYPRE_StructVectorInitialize(b);

    // for (i = 0; i < 18; i++)
    // {
    //     values_b[i] = 0.0;
    // }

    // HYPRE_StructVectorSetBoxValues(b, ilower[0], iupper[0], values_b);
    // HYPRE_StructVectorSetBoxValues(b, ilower[1], iupper[1], values_b);

    // HYPRE_StructVectorAssemble(b);

    /* Free memory */
    HYPRE_StructGridDestroy(grid);
    HYPRE_StructStencilDestroy(stencil);
    HYPRE_StructMatrixDestroy(A);
    // HYPRE_StructVectorDestroy(b);

    // free(values);
    cudaFree(values);
    free(stencil_indices);
    free(ilower[0]);
    free(ilower[1]);
    free(ilower);
    free(iupper[0]);
    free(iupper[1]);
    free(iupper);

    for (int i = 0; i < size; i++)
        free(offsets[i]);

    free(offsets);

    /* Finalize Hypre */
    HYPRE_Finalize();

    /* Finalize MPI */
    MPI_Finalize();

    return 0;
}
