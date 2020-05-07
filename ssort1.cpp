// Parallel sample sort
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <algorithm>

int main( int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  // Number of random numbers per processor (this should be increased
  // for actual tests or could be passed in through the command line
  int N = 10000;
  int* vec = (int*)malloc(N*sizeof(int));
  // seed random number generator differently on every core
  srand((unsigned int) (rank + 393919));



  // fill vector with random integers
  for (int i = 0; i < N; ++i) {
    vec[i] = rand();
  }
  printf("rank: %d, first entry: %d\n", rank, vec[0]);
  


  // sort locally
  double t = MPI_Wtime();
  std::sort(vec, vec+N);
  //for(long i =0; i<N;i++){printf("%d rank %d\n",vec[i],rank);}


  // sample p-1 entries from vector as the local splitters, i.e.,
  // every N/P-th entry of the sorted vector
  int* sample_loc = (int *) malloc((p-1)*sizeof(int));
  int c = 0;
  for (int j=0; j<N && c<(p-1); j++){
	if((j+1)%(N/p) == 0){sample_loc[c] = vec[j]; c = c+1;}
  }
  //if(0==rank){printf("splitters \n\n");}
  //for(long i =0; i<p-1;i++){printf("%d rank %d\n",sample_loc[i],rank);}


  
  // every process communicates the selected entries to the root
  // process; use for instance an MPI_Gather
  int* root_sam_recv = NULL;
  if (0 == rank){
	root_sam_recv = (int *) malloc(p*(p-1)*sizeof(int)); 
  }
  MPI_Gather(sample_loc, p - 1, MPI_INT, root_sam_recv, p - 1, MPI_INT, 0, MPI_COMM_WORLD);
  //if(0==rank){printf("\n\ngathered\n\n");for(long i =0; i<(p*(p-1));i++){printf("%d rank %d\n",root_sam_recv[i],rank);}}



  // root process does a sort and picks (p-1) splitters (from the
  // p(p-1) received elements)
  int* root_sam_pick = (int *) malloc((p-1)*sizeof(int));;
  if(0 == rank){
	std::sort(root_sam_recv, root_sam_recv+(p*(p-1)));
	int c = 0;
	for(int j=0; j<p*(p-1) && c<p-1; j++){
		if((j+1)%p==0){root_sam_pick[c]= root_sam_recv[j] ; c=c+1;}
	}
  }  
  //if(0==rank){printf("\n\n gathered and sorted and picked p-1 \n\n");for(long i =0; i<(p-1);i++){printf("%d rank %d\n",root_sam_pick[i],rank);}}
  


    
  // root process broadcasts splitters to all other processes
  MPI_Bcast(root_sam_pick, p - 1, MPI_INT, 0, MPI_COMM_WORLD);
  //MPI_Barrier(MPI_COMM_WORLD);
  //printf("\n\n");
  //for(long i =0; i<(p-1);i++){printf("%d rank %d\n",root_sam_pick[i],rank);}printf("\n\n");
 


  // every process uses the obtained splitters to decide which
  // integers need to be sent to which other process (local bins).
  // Note that the vector is already locally sorted and so are the
  // splitters; therefore, we can use std::lower_bound function to
  // determine the bins efficiently.
  //
  // Hint: the MPI_Alltoallv exchange in the next step requires
  // send-counts and send-displacements to each process. Determining the
  // bins for an already sorted array just means to determine these
  // counts and displacements. For a splitter s[i], the corresponding
  // send-displacement for the message to process (i+1) is then given by,
  // sdispls[i+1] = std::lower_bound(vec, vec+N, s[i]) - vec;
    
  int* s1 = (int*) malloc(p * sizeof(int)); 
  int* r1 = (int*) malloc(p * sizeof(int)); 
  int* bin_send = (int*) malloc(p * sizeof(int));
  int* bin_recv = (int*) malloc(p * sizeof(int)); 
  for (int j = 0; j < p-1; j++){
 	 if(j==0){ 
		s1[0] =0; r1[0] = 0;
	 }
    	 s1[j+1] = std::lower_bound(vec, vec+N, root_sam_pick[j]) - vec;
	 if(j==(p-2)){bin_send[p-1] = N - s1[p-1];}
	 bin_send[j]=s1[j+1]-s1[j];
  }


  // send and receive: first use an MPI_Alltoall to share with every
  // process how many integers it should expect, and then use
  // MPI_Alltoallv to exchange the data
  MPI_Alltoall(bin_send, 1, MPI_INT, bin_recv, 1, MPI_INT, MPI_COMM_WORLD);
  for (int  i = 0; i < p-1; i++) {
	r1[i+1] = r1[i] + bin_recv[i];  
  }
  int* final_sort = (int*) malloc((r1[p-1] + bin_recv[p-1]) * sizeof(int));
  MPI_Alltoallv(vec, bin_send, s1, MPI_INT, final_sort, bin_recv, r1, MPI_INT, MPI_COMM_WORLD);
 


  // do a local sort of the received data
  std::sort(final_sort, final_sort + r1[p-1] + bin_recv[p-1]);
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
	printf("Elapsed time (s) : %lf\n", t = MPI_Wtime()-t);
  }

  
  FILE* f1 = NULL;
  char filename[256];
  snprintf(filename, 256, "output%02d.txt", rank);
  f1 = fopen(filename,"w+");

  if(!f1) {
    printf("Error opening file \n");
    return 1;
  }
  
  for(int n = 0; n < (r1[p-1] + bin_recv[p-1]); ++n){
    fprintf(f1, "%d\n", final_sort[n]);
  }
  fclose(f1);  
  free(vec);
  free(s1);
  free(bin_send);
  free(r1);
  free(bin_recv);
  free(root_sam_pick);
  free(root_sam_recv);
  free(sample_loc);
  free(final_sort);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
