/* MPI-parallel Jacobi smoothing to solve -u''=f
 * Global vector has N unknowns, each processor works with its
 * part, which has lN = N/p unknowns.
 * Author: Georg Stadler
 */
//mpirun <exec> <N> <max_iters> <lN>

#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

/* compuate global residual, assuming ghost values are updated */
double compute_residual(double *lu, int lN, double invhsq){
  int i;
  double tmp, gres = 0.0, lres = 0.0;

  for(long j =1; j<=lN;j++){
  	for (i = 1; i <= lN; i++){
    		tmp = ((4.0*lu[i+(lN+2)*j] - lu[(i-1)+(lN+2)*j] - lu[(i+1)+(lN+2)*j] - lu[i+(lN+2)*(j-1)] - lu[i+(lN+2)*(j+1)]))*invhsq -1;
    		lres += tmp * tmp;
	}
  }

  /* use allreduce for convenience; a reduce would also be sufficient */
  MPI_Allreduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return sqrt(gres);
}


int main(int argc, char * argv[]){
  int mpirank, i,j, p, N, lN, iter, max_iters;
  MPI_Status status, status1,status2,status3;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  /* get name of host running MPI process */
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Rank %d/%d running on %s.\n", mpirank, p, processor_name);

  sscanf(argv[1], "%d", &N);
  sscanf(argv[2], "%d", &max_iters);
  sscanf(argv[3], "%d", &lN);

  /* timing */
  
  MPI_Barrier(MPI_COMM_WORLD);
  double tt = MPI_Wtime();

  /* Allocation of vectors, including left/upper and right/lower ghost points */
  double *lu    = (double *) calloc( sizeof(double),(lN+2)*(lN+2));
  double *lunew = (double *) calloc( sizeof(double),(lN+2)*(lN+2));
  double *f = (double *) calloc( sizeof(double),(lN+2)*(lN+2));
  double *lutemp;
  double h = 1.0 / (N + 1);
  double hsq = h * h;
  double invhsq = 1./hsq;
  double gres, gres0, tol = 1e-5;
  int jump = N/lN;
 
  for (long k=0; k<(lN+2)*(lN+2);k++){
	f[k]=1.0*hsq;
  }

  /* initial residual */
  gres0 = compute_residual(lu, lN, invhsq);
  gres = gres0;
  for (iter = 0; iter < max_iters && gres/gres0 > tol; iter++) {

    /* Jacobi step for local points */ 
    for (long j = 1; j<=lN; j++){
    	for (long i = 1; i <= lN; i++){
      		lunew[i+j*(lN+2)]  = 0.25 * (f[j] + lu[(i-1)+j*(lN+2)] +lu[(i+1)+j*(lN+2)]+lu[i+(j-1)*(lN+2)]+lu[i+(j+1)*(lN+2)]);
	}
    }
 

    /* communicate ghost values */
    for (long k = 1; k<=lN; k++){
     if ((mpirank+1)%jump !=0) {
      // If not the last process, send/recv bdry values to the right 
      	MPI_Send(&(lunew[lN + k*(lN+2)]), 1, MPI_DOUBLE, mpirank+1, 124, MPI_COMM_WORLD);
      	MPI_Recv(&(lunew[(lN+1) + k*(lN+2)]), 1, MPI_DOUBLE, mpirank+1, 123, MPI_COMM_WORLD, &status);
        
	}
    

    if (mpirank != 0 && mpirank % jump !=0 ) {
     //  If not the first process, send/recv bdry values to the left 
      	MPI_Send(&(lunew[1+k*(lN+2)]), 1, MPI_DOUBLE, mpirank-1, 123, MPI_COMM_WORLD);
      	MPI_Recv(&(lunew[0+k*(lN+2)]), 1, MPI_DOUBLE, mpirank-1, 124, MPI_COMM_WORLD, &status1);
      }
    

    if (mpirank > jump-1) {
      // If not the last process, send/recv bdry values to the upper part 
        MPI_Send(&(lunew[k + (lN+2)*1]), 1, MPI_DOUBLE, mpirank-jump, 234, MPI_COMM_WORLD);
        MPI_Recv(&(lunew[k + (lN+2)*0]), 1, MPI_DOUBLE, mpirank-jump, 133, MPI_COMM_WORLD, &status2);
      }
    


    if (mpirank < (jump*jump)-jump ) {
     //  If not the first process, send/recv bdry values to the lower part 
        MPI_Send(&(lunew[k+(lN+1)*(lN+2)]), 1, MPI_DOUBLE, mpirank+jump, 133, MPI_COMM_WORLD);
        MPI_Recv(&(lunew[k+lN*(lN+2)]), 1, MPI_DOUBLE, mpirank+jump, 234, MPI_COMM_WORLD, &status3);
      }MPI_Barrier(MPI_COMM_WORLD);
    }


    /* copy newu to u using pointer flipping */
    lutemp = lu; lu = lunew; lunew = lutemp;
    if (0 == (iter % 10)) {
      gres = compute_residual(lu, lN, invhsq);
      if (0 == mpirank) {
	printf("Iter %d: Residual: %g\n", iter, gres);
      }
    }
  }

  /* Clean up */
  free(lu);
  free(lunew);
  free(f);

  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = MPI_Wtime() - tt;
  if (0 == mpirank) {
    printf("Time elapsed is %f seconds.\n", elapsed);
  }
  MPI_Finalize();
  return 0;
}
