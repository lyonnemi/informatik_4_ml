import random
import math
import matplotlib.pyplot as plt



def sigmoid(x):
    # input: z is a number or array
    # task: return 1 / (1 + exp(-z))
    # hint: for stability you can use the if z>=0 else form
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        return math.exp(x) / (1 + math.exp(x))

def cost(y_true, y_pred):
    # Handle both single values and lists
    if isinstance(y_true, (list, tuple)):
        m = len(y_true)
        cost_value = 0.0
        for i in range(m):
            cost_value += y_true[i] * math.log(max(y_pred[i], 1e-15)) + (1 - y_true[i]) * math.log(max(1 - y_pred[i], 1e-15))
        cost_value = -(1 / m) * cost_value
    else:
        # Single value case
        cost_value = -(y_true * math.log(max(y_pred, 1e-15)) + (1 - y_true) * math.log(max(1 - y_pred, 1e-15)))
    return cost_value
    


def forward(input1, input2, p):
    # input:
    #   input1 (int: 0 or 1)
    #   input2 (int: 0 or 1)
    #   p (dict) 
    # output:
    #   y_pred
    #   saved (dict) must include at least: input1, input2, h1, h2, y_pred

    h1 = sigmoid(p["theta_11_1"] * input1 + p["theta_12_1"] * input2 + p["a_0_1"])
    h2 = sigmoid(p["theta_21_1"] * input1 + p["theta_22_1"] * input2 + p["a_0_2"])
    y_pred = sigmoid(p["theta_10_1"] * h1 + p["theta_20_1"] * h2)

    saved = {
        "input1": input1,
        "input2": input2,
        "h1": h1,
        "h2": h2,
        "y_pred": y_pred
    }

    return y_pred, saved


def backward(y_true, saved, p):
    # input:
    #   y_true (int: 0 or 1)
    #   saved (dict) from forward(): input1, input2, h1, h2, y_pred
    #   p (dict) current parameters
    # output:
    #   grads (dict) gradients with SAME keys as p
    
    input1 = saved["input1"]
    input2 = saved["input2"]
    
    grads = {k: 0.0 for k in p}
    # Compute gradients using backpropagation
    # For binary cross-entropy with sigmoid: dL/dy_pred = (y_pred - y_true) / (y_pred * (1 - y_pred))
    # But we need dL/dz where z is the pre-activation, so: dL/dz = y_pred - y_true
    # Then multiply by sigmoid derivative for hidden layer
    dL_dz_out = saved["y_pred"] - y_true
    grads["theta_10_1"] = dL_dz_out * saved["h1"]
    grads["theta_20_1"] = dL_dz_out * saved["h2"]
    # Hidden layer gradients: multiply by sigmoid derivative
    dL_dh1 = dL_dz_out * p["theta_10_1"]
    dL_dh2 = dL_dz_out * p["theta_20_1"]
    grads["theta_11_1"] = dL_dh1 * saved["h1"] * (1 - saved["h1"]) * input1
    grads["theta_12_1"] = dL_dh1 * saved["h1"] * (1 - saved["h1"]) * input2
    grads["theta_21_1"] = dL_dh2 * saved["h2"] * (1 - saved["h2"]) * input1
    grads["theta_22_1"] = dL_dh2 * saved["h2"] * (1 - saved["h2"]) * input2
    grads["a_0_1"] = dL_dh1 * saved["h1"] * (1 - saved["h1"])
    grads["a_0_2"] = dL_dh2 * saved["h2"] * (1 - saved["h2"])


    return grads
    

def apply_updates(p, sum_grads, learning_rate, n_samples):

    # input:
    #   p (dict) parameters
    #   sum_grads (dict) summed gradients over all samples
    #   learning_rate
    #   n_samples
    # output:
    #   update p in-place:
    for k in p:
        p[k] -= learning_rate * sum_grads[k] / n_samples


def random_params(seed):
    random.seed(seed)
    p = {}
    p["theta_11_1"] = random.uniform(-1.0, 1.0)
    p["theta_12_1"] = random.uniform(-1.0, 1.0)
    p["theta_21_1"] = random.uniform(-1.0, 1.0)
    p["theta_22_1"] = random.uniform(-1.0, 1.0)

    p["theta_11_2"] = random.uniform(-1.0, 1.0)
    p["theta_12_2"] = random.uniform(-1.0, 1.0)

    p["theta_10_1"] = random.uniform(-1.0, 1.0)
    p["theta_20_1"] = random.uniform(-1.0, 1.0)
    p["theta_10_2"] = random.uniform(-1.0, 1.0)

    p["a_0_1"] = 1.0
    p["a_0_2"] = 1.0

    return p


def train(inputs, targets, learning_rate, epochs, seed):
    p = random_params(seed)
    costs = []

    for epoch in range(epochs):
        sum_grads = {k: 0.0 for k in p}
        total_cost = 0.0

        for i in range(len(inputs)):
            input1, input2 = inputs[i]
            y_true = targets[i]

            y_pred, saved = forward(input1, input2, p)
            total_cost += cost(y_true, y_pred)

            grads = backward(y_true, saved, p)
            for k in sum_grads:
                sum_grads[k] += grads[k]

        avg_cost = total_cost / len(inputs)
        costs.append(avg_cost)

        apply_updates(p, sum_grads, learning_rate, len(inputs))

        if epoch % 1000 == 0:
            print(f"epoch {epoch:4} | cost {avg_cost}")

    return p, costs


def show_results(name, p, inputs, targets):
    print("\n" + name)
    print(f"{'input1':<6} {'input2':<6} {'h1':<10} {'h2':<10} {'y_pred':<10} {'y_class':<5} {'y_true':<5}")
    for i in range(len(inputs)):
        input1, input2 = inputs[i]
        y_true = targets[i]
        y_pred, saved = forward(input1, input2, p)
        y_class = 1 if y_pred >= 0.5 else 0
        print(f"{input1:<6d} {input2:<6d} {saved['h1']:<10.5f} {saved['h2']:<10.5f} {y_pred:<10.5f} {y_class:<6d} {y_true:<6d}")

    print("\nParameters")
    for k in p:
        print(f"{k:12}= {p[k]}")
    print("\n")


def plot_loss(costs, title):
    plt.plot(costs)
    plt.xlabel("epochs")
    plt.ylabel("cost")
    plt.title(title)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    targets_and = [0, 0, 0, 1]
    targets_xor = [0, 1, 1, 0]

    learning_rate = 0.1
    epochs = 10000

    p_and, cost_and = train(inputs, targets_and, learning_rate, epochs, seed=1)
    show_results("AND results", p_and, inputs, targets_and)
    plot_loss(cost_and, "AND")

    p_xor, cost_xor = train(inputs, targets_xor, learning_rate, epochs, seed=2)
    show_results("XOR results", p_xor, inputs, targets_xor)
    plot_loss(cost_xor, "XOR")