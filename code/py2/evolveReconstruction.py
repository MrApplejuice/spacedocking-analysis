from PyGMO import *
from pickle import *
from reconstructionProblem import *
import numpy as np

def evolveReconstruction(filename = 'test', n_views=2, n_points=100, IPs=[], t_scale = 3.0, world_scale=10.0, K = [], x = []):
	
	sd = 1234;
	prob = reconstructionProblem(n_views, n_points, IPs, t_scale, world_scale, K);
	n_genes = prob.getGenomeSize();
	print 'n_genes = %d' % n_genes
	
	# all values are scaled afterward:
	prob.lb = [-1.0] * n_genes;
	prob.ub = [1.0] * n_genes;

	# evolutions, generations and individuals
	n_evolutions = 1;
	n_generations = 1;
	n_individuals = 1;

	if(len(x) == 0):
		# prior on camera matrices and points:
		for v in range(n_views-1):
			x = x + [0.0] * 6;		
		for p in range(n_points):
			x = x + [0.0, 0.5, 0.0];
	else:
		print 'Prior genome: length = %d, max = %f, min = %f' % (len(x), np.max(np.array(x)), np.min(np.array(x)));

	print 'Creating algorithms'
	algo_list = []
	# algo_list.append(algorithm.monte_carlo(iter = n_generations))
	# algo_list.append(algorithm.py_cmaes(gen = n_generations))
	# algo_list.append(algorithm.de(gen = n_generations))
	
	#algo_list.append(algorithm.pso_gen(gen = n_generations, vcoeff = 0.2));
	#algo_list.append(algorithm.pso_gen(gen = n_generations, vcoeff = 0.1));
	# algo_list.append(algorithm.pso_gen(gen = n_generations, vcoeff = 0.2));
	# algo_list.append(algorithm.pso_gen(gen = n_generations, vcoeff = 0.2));
	#algo_list.append(algorithm.pso_gen(gen = n_generations, variant=3))
	#algo_list.append(algorithm.pso_gen(gen = n_generations, variant=4))
	#algo_list.append(algorithm.pso_gen(gen = n_generations, variant=5))
	#algo_list.append(algorithm.pso_gen(gen = n_generations, variant=6))
	#algo_list.append(algorithm.pso_gen(gen = n_generations, variant=5, neighb_type=1))
	#algo_list.append(algorithm.pso_gen(gen = n_generations, variant=5, neighb_type=3))
	#algo_list.append(algorithm.pso_gen(gen = n_generations, variant=5, neighb_type=4))
	#algo_list.append(algorithm.pso_gen(gen = n_generations, variant=6, neighb_type=1))
	#algo_list.append(algorithm.pso_gen(gen = n_generations, variant=6, neighb_type=3))
	#algo_list.append(algorithm.pso_gen(gen = n_generations, variant=6, neighb_type=4))
	
	algo_list.append(algorithm.scipy_tnc(maxfun=150));

	#algo_list.append(algorithm.sga(gen = n_generations, cr = 0, m = 0.05))
	# algo_list.append(algorithm.sga(gen = n_generations, cr = 0.05, m = 0.05))
	# algo_list.append(algorithm.sga(gen = n_generations, cr = 0.05, m = 0.01))
	#algo_list.append(algorithm.sga(gen = n_generations, cr = 0.1, m = 0.03))
	#algo_list.append(algorithm.sga(gen = n_generations, cr = 0.5, m = 0.5))
	#algo_list.append(algorithm.sga(gen = n_generations, cr = 0.005, m = 0.001))
	
	#algo_list.append(algorithm.sga(gen = n_generations, cr = 0.1, m = 0.03, selection = algorithm._algorithm._selection_type.BEST20))
	#algo_list.append(algorithm.sga(gen = n_generations, cr = 0.1, m = 0.03, selection = algorithm._algorithm._selection_type.BEST20))
	#algo_list.append(algorithm.sga(gen = n_generations, cr = 0.1, m = 0.03, selection = algorithm._algorithm._selection_type.BEST20))
	
	#algo_list.append(algorithm.sga(gen = n_generations, cr = 0.1, m = 0.03, selection = algorithm._algorithm._selection_type.BEST20))
	#algo_list.append(algorithm.sga(gen = n_generations, cr = 0.1, m = 0.03, selection = algorithm._algorithm._selection_type.BEST20))

	topo = topology.ring()

	print 'Creating archipelago'
	archi = archipelago(topology = topo)
	# len(algo_list)
	
	n_islands = len(algo_list);

	for i in range(0, n_islands):
			print 'Island %d' % i
			if(len(x) == 0):
				archi.push_back(island(algo_list[i], prob, n_individuals))
			else:
				print 'starting with prior!'
				pop = population(prob);
				for ind in range(n_individuals):
					pop.push_back(x);
				archi.push_back(island(algo_list[i], pop))
			print 'Done with push back'

	avg_fits = np.zeros([n_islands,1]);
	
	for evs in range(0,n_evolutions):
		print '****************'
		print 'Evolving step %d' % evs
		print '****************'
		archi.evolve(1);
		archi.join()
		
		# save results until now
		# show the results for this step:
		i = 0
		isl_num = 1
		best_fit = 1E8;
		best_ind = 0;
		for isl in archi:
			f = open('genome_'+filename+'_'+str(isl_num), "a")
			f2 = open('fitness'+filename+'_'+str(isl_num), "a")	
			champ = isl.population.champion;
			isl_num = isl_num + 1
			print 'Isl. %d' % i,
			print 'fit %3.3f ' % champ.f,
			if(champ.f < best_fit):
				best_fit = champ.f;
				best_ind = i;
			if(evs == 0):
				avg_fits[i] = champ.f;
			else:
				avg_fits[i] = avg_fits[i] * 0.9 + champ.f[0] * 0.1;
			print 'avg_fit %3.3f' % avg_fits[i],
			
			# only numbers with spaces so that we can easily import them automatically afterwards:
			#f.write('x = array([')
			j = 1
			for ss in champ.x:
				f.write("".join(str(ss)))
				if j < len(champ.x):
					f.write(' ') #(',')
				j = j + 1
			#f.write('])')
			f.write('\n')
			f2.write("".join(str(champ.f)))
			f2.write('\n')
			i += 1
		
			f.close()
			f2.close()
		print ' '

	# return the best individual:
	genome = archi[best_ind].population.champion.x;
	(Rs, Ts, X) = prob.transformGenomeToMatrices(genome, n_views, n_points);
	return (Rs, Ts, X);
	

def constructGenome(phis, thetas, psis, Ts, n_world_points, Xs=[]):
	""" Uses the information present from for example the drone's state estimation to 
			initialize the genome in a sensible way. 
	"""

	world_scale = 10.0;
	t_scale = 3.0;

	# phi, theta, psi, translation of the first view should be all zeros	
	n_views = len(phis);
	genome = [];
	for v in range(n_views):
		t = Ts[v].tolist();
		genome += [phis[v] / np.pi, thetas[v] / np.pi, psis[v] / np.pi, t[0][0] / t_scale, t[1][0] / t_scale, t[2][0] / t_scale];

	if(len(Xs) == 0):
		for p in range(n_world_points):
			genome += [0.0, 0.0, 5.0 / world_scale];
	else:
		for p in range(n_world_points):
			genome += (Xs[p,:] / world_scale).tolist();

	# ensure bounds:
	for g in range(len(genome)):
		gene = genome[g];
		if(gene >= 1.0):
			genome[g] = 0.9999;
		elif(gene <= -1.0):
			genome[g] = -0.9999;
	
	return genome;

