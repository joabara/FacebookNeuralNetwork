/**
 * The main class that handles the entire network
 * Has multiple attributes each with its own use
 *
 */

import java.util.*;


public class NNImpl
{
	public ArrayList<Node> inputNodes=null;//list of the output layer nodes.
	public ArrayList<Node> hiddenNodes=null;//list of the hidden layer nodes
	public Node outputNode=null;// single output node that represents the result of the regression

	public ArrayList<Instance> trainingSet=null;//the training set

	Double learningRate=1.0; // variable to store the learning rate
	int maxEpoch=1; // variable to store the maximum number of epochs


	/**
 	* This constructor creates the nodes necessary for the neural network
 	* Also connects the nodes of different layers
 	* After calling the constructor the last node of both inputNodes and
 	* hiddenNodes will be bias nodes.
 	*/

	public NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch, Double [][]hiddenWeights, Double[] outputWeights)
	{
		this.trainingSet=trainingSet;
		this.learningRate=learningRate;
		this.maxEpoch=maxEpoch;

		//input layer nodes
		inputNodes=new ArrayList<Node>();
		int inputNodeCount=trainingSet.get(0).attributes.size();
		for(int i=0;i<inputNodeCount;i++)
		{
			Node node=new Node(0);
			inputNodes.add(node);
		}

		//bias node from input layer to hidden
		Node biasToHidden=new Node(1);
		inputNodes.add(biasToHidden);

		//hidden layer nodes
		hiddenNodes=new ArrayList<Node> ();
		for(int i=0;i<hiddenNodeCount;i++)
		{
			Node node=new Node(2);
			//Connecting hidden layer nodes with input layer nodes
			for(int j=0;j<inputNodes.size();j++)
			{
				NodeWeightPair nwp=new NodeWeightPair(inputNodes.get(j),hiddenWeights[i][j]);
				node.parents.add(nwp);
			}
			hiddenNodes.add(node);
		}

		//bias node from hidden layer to output
		Node biasToOutput=new Node(3);
		hiddenNodes.add(biasToOutput);



		Node node=new Node(4);
		//Connecting output node with hidden layer nodes
		for(int j=0;j<hiddenNodes.size();j++)
		{
			NodeWeightPair nwp=new NodeWeightPair(hiddenNodes.get(j), outputWeights[j]);
			node.parents.add(nwp);
		}
		outputNode = node;

	}

	/**
	 * Get the output from the neural network for a single instance. That is, set the values of the training instance to
	the appropriate input nodes, percolate them through the network, then return the activation value at the single output
	node. This is your estimate of y.
	 */

	public double calculateOutputForInstance(Instance inst)
	{
		// get inputs from instance caputred in inputNodes
		for(int k = 0; k < inst.attributes.size(); k++)
		{
			inputNodes.get(k).setInput(inst.attributes.get(k));
			inputNodes.get(k).calculateOutput();
		}

		ArrayList<Double> inputs = inst.attributes;
		Double sum = 0.0;

		// for each hidden Nodes in layer
		for(Node perceptron : this.hiddenNodes)
		{
			if(perceptron != null && perceptron.parents!=null)
			{
				// compute the sum of the inputs and weights
				for(int i =0; i < inputs.size(); i++)
				{
					Double xi = inputs.get(i);

					Double wi = perceptron.parents.get(i).weight;

					sum += xi*wi;
				}
				perceptron.setInput(sum);
				perceptron.calculateOutput();
			}
		}

		outputNode.calculateOutput();

		return outputNode.getOutput();
	}





	/**
	 * Trains a neural network with the parameters initialized in the constructor for the number of epochs specified in the instance variable maxEpoch.
	 * The parameters are stored as attributes of this class, namely learningRate (alpha) and trainingSet.
	 * Implement stochastic graident descent: update the network weights using the deltas computed after each the error of each training instance is computed.
	 * An single epoch looks at each instance training set once, so you should update weights n times per epoch if you have n instances in the training set.
	 */

	public void train()
	{
		int epoch = 0; // start epoch

		// complete until epoch limit reached
		while(epoch < this.maxEpoch)
		{
			for(Instance ex : trainingSet)
			{
				//reset inputNodes
				for(int in = 0; in < ex.attributes.size(); in++)
				{
					inputNodes.get(in).setInput(ex.attributes.get(in));
					inputNodes.get(in).calculateOutput();
				}

				// error computation
				Double o = calculateOutputForInstance(ex);
				Double t = ex.output;
				Double error = t-o;

				// compute delta W-jk
				ArrayList<Double> deltaWeightJK = getWeightsJK(error);

				for(int k = 0; k < deltaWeightJK.size(); k++)
					outputNode.parents.get(k).weight += deltaWeightJK.get(k);

				// compute delta W-ij
				for(int i = 0; i < hiddenNodes.size()-1; i++)
				{
					if(hiddenNodes.get(i) != null)
					{
						ArrayList<Double> deltaWeightIJ = getWeightsIJ(error,i);

						for(int j = 0; j < hiddenNodes.get(i).parents.size(); j++)
							hiddenNodes.get(i).parents.get(j).weight += deltaWeightIJ.get(j);
					}
				}
			}
			epoch++;
		}
	}

	private ArrayList<Double> getWeightsJK(Double error)
	{
		// alpha = learningRate
		// aj output of hiddenNodes
		// (T - O) = error
		// g'(x) = gPrime(x)

		ArrayList<Double> deltaWeightJK = new ArrayList<Double>();
		Double alpha = this.learningRate;

		for(int j = 0; j < hiddenNodes.size(); j++)
		{
			Node xj = hiddenNodes.get(j);
			Double aj = xj.getOutput();
			Double gP = gPrime(outputNode.getSum());

			// computing delta wjk
			Double deltaWjk = alpha*aj*error*gP;
			deltaWeightJK.add(deltaWjk);
		}

		return deltaWeightJK;
	}

	// get weights of input to hidden by hiddeni
	private ArrayList<Double> getWeightsIJ(Double error, int i)
	{
		// alpha = learningRate
		// ai output of inputNodes
		// (T - O) = error
		// g'(x) = gPrime(x)

		ArrayList<Double> deltaWeightIJ = new ArrayList<Double>();
		Double alpha = this.learningRate;

		for(int j = 0; j < this.hiddenNodes.get(i).parents.size(); j++)
		{
			Double ai = this.hiddenNodes.get(i).parents.get(j).node.getOutput();

			Double gP = gPrime(hiddenNodes.get(i).getSum());
			ArrayList<Double> deltaWeightJK = getWeightsJK(error);

			deltaWeightJK = multiplyVector(deltaWeightJK, error);
			deltaWeightJK = multiplyVector(deltaWeightJK, gPrime(outputNode.getSum()));

			// computing delta Wij
			Double deltaWij = alpha*ai*gP*sumOfList(deltaWeightJK);

			deltaWeightIJ.add(deltaWij);
		}

		return deltaWeightIJ;
	}

	// gPrime function from the assignment description
	private double gPrime(Double x)
	{
		if(x > 0.0) return 1.0;
		return 0.0;
	}


	private Double sumOfList(ArrayList<Double> list)
	{
		double sum = 0.0;

		for(Double value : list)
			sum+=value;

		return sum;
	}

	private ArrayList<Double> multiplyVector(ArrayList<Double> list, Double constant)
	{
		for(Double value : list)
		{
			value = value * constant;
		}

		return list;
	}

	/**
	 * Returns the mean squared error of a dataset. That is, the sum of the squared error (T-O) for each instance
	in the dataset divided by the number of instances in the dataset.
	 */
	public double getMeanSquaredError(List<Instance> dataset)
	{
		double sse = 0.0;

		for(Instance ex : dataset)
		{
			Double o = calculateOutputForInstance(ex);
			Double t = ex.output;
			Double error = (t-o)*(t-o);
			sse+=error;
		}

		return sse/dataset.size();
	}
}
