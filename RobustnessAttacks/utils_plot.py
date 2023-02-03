import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as ssim
from PIL import Image


def save_adversarial_image(original, perturbed, label, prediction, index, dir_to_folder):
    """
    displays image, adversarial image, difference image and each of the color channels of the difference image
    :param original: a torch tensor on cpu
    :param perturbed: a torch tensor on cpu
    :param title: the title of the plot
    """
    original = np.moveaxis(original.detach().cpu().numpy(), 0, -1)
    perturbed = np.moveaxis(perturbed.detach().cpu().numpy(), 0, -1)

    # difference = original - attacked
    difference = perturbed - original
    diff_gray = np.dot(difference[..., :3], [0.299, 0.587, 0.144])

    # sim = ssim(original, attacked, multichannel=True, full=True)

    original_gray = np.dot(original[..., :3], [0.299, 0.587, 0.144])
    attacked_gray = np.dot(perturbed[..., :3], [0.299, 0.587, 0.144])

    sim = ssim(original_gray, attacked_gray, full=True)
    # print('sim: ', sim[0])

    # display the images
    title = f"idx: {index}, label: {label}, prediction: {prediction.item()}, ssim = {sim[0]}"
    fig = plt.figure('Adversarial example: ' + title)
    plt.suptitle(f"Adversairal example: \n {title}")
    a = fig.add_subplot(2, 3, 1)
    plt.imshow(original)
    a.set_title('original')

    a = fig.add_subplot(2, 3, 2)
    plt.imshow(sim[1], cmap='gray')
    a.set_title('difference image')

    a = fig.add_subplot(2, 3, 3)
    plt.imshow(perturbed)
    a.set_title('adversarial image')

    # choose best cmaps (Reds, Greens, Blues) (gray) (binary)
    a = fig.add_subplot(2, 3, 4)
    plt.imshow(difference[..., 0], cmap='Reds')
    a.set_title('Red')

    a = fig.add_subplot(2, 3, 5)
    plt.imshow(difference[..., 1], cmap='Greens')
    a.set_title('Green')

    a = fig.add_subplot(2, 3, 6)
    plt.imshow(difference[..., 2], cmap='Blues')
    a.set_title('Blue')

    plt.savefig(fname=f"{dir_to_folder}idx {index}, label {label}, prediction {prediction.item()}.png")
    plt.close('all')


def show_mask_results(original, attacked, masked):
    original = np.moveaxis(original.detach().cpu().numpy(), 0, -1)
    attacked = np.moveaxis(attacked.detach().cpu().numpy(), 0, -1)
    masked = np.moveaxis(masked.detach().cpu().numpy(), 0, -1)

    # display the images
    fig = plt.figure('Adversarial example')
    a = fig.add_subplot(1, 3, 1)
    plt.imshow(original)
    a.set_title('original')

    a = fig.add_subplot(1, 3, 2)
    plt.imshow(masked)
    a.set_title('masked image')

    a = fig.add_subplot(1, 3, 3)
    plt.imshow(attacked)
    a.set_title('adversarial image')
    plt.show()


def show_adversarial_images(original, attacked, title):
    """
    displays image, adversarial image, difference image and each of the color channels of the difference image
    :param original: a torch tensor on cpu
    :param attacked: a torch tensor on cpu
    :param title: the title of the plot
    """
    original = np.moveaxis(original.detach().cpu().numpy(), 0, -1)
    attacked = np.moveaxis(attacked.detach().cpu().numpy(), 0, -1)

    # difference = original - attacked
    difference = attacked - original
    diff_gray = np.dot(difference[..., :3], [0.299, 0.587, 0.144])

    # sim = ssim(original, attacked, multichannel=True, full=True)

    original_gray = np.dot(original[..., :3], [0.299, 0.587, 0.144])
    attacked_gray = np.dot(attacked[..., :3], [0.299, 0.587, 0.144])
    sim = ssim(original_gray, attacked_gray, full=True)
    print('sim: ', sim[0])

    # display the images
    fig = plt.figure('Adversarial example: ' + title)
    a = fig.add_subplot(2, 3, 1)
    plt.imshow(original)
    a.set_title('original')

    a = fig.add_subplot(2, 3, 2)
    plt.imshow(sim[1], cmap='gray')
    a.set_title('difference image')

    a = fig.add_subplot(2, 3, 3)
    plt.imshow(attacked)
    a.set_title('adversarial image')

    # TODO choose best cmaps (Reds, Greens, Blues) (gray) (binary)
    a = fig.add_subplot(2, 3, 4)
    plt.imshow(difference[..., 0], cmap='Reds')
    a.set_title('Red')

    a = fig.add_subplot(2, 3, 5)
    plt.imshow(difference[..., 1], cmap='Greens')
    a.set_title('Green')

    a = fig.add_subplot(2, 3, 6)
    plt.imshow(difference[..., 2], cmap='Blues')
    a.set_title('Blue')

    plt.show()


def plot_eps_vs_accuracy(epsilons, accuracies, attacks, modeltype_names):
    """
    Deprecated.
    :param epsilons:
    :param accuracies:
    :param attacks:
    :param modeltype_names:
    :return:
    """
    print()
    linetype = ['-', '--', ':', '-.']
    linecolors = ['b', 'r', 'g', 'y', 'm', 'c', 'k', 'w']
    fig, ax = plt.subplots()

    for i in range(len(modeltype_names)):
        for j in range(len(attacks)):
            seed1, seed2, seed3 = accuracies[i]
            acc = np.stack([seed1[j], seed2[j], seed3[j]])
            mean = np.mean(acc, axis=0)
            std = np.std(acc, axis=0)

            print()
            print('model: ', modeltype_names[i])
            print('mean: ', mean)
            print('std: ', std)

            ax.plot(epsilons, mean, "*-", label=modeltype_names[i] + ' ' + attacks[j],
                    linestyle=linetype[i], color=linecolors[j])
            ax.fill_between(epsilons, mean - std, mean + std, color=linecolors[j], alpha=0.5)
            plt.yticks(np.arange(0, 1.1, step=0.1))
            # plt.xticks(np.arange(0, .35, step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def plot_eps_vs_accuracy_multiple(epsilons, accuracies, attacks, modeltype_names, normalize=False, use_seaborn=False,
                                  plt_colors=[], fontsize='medium'):
    linetype = ['-', '--', ':', '-.']
    if len(plt_colors) > 0 and len(plt_colors) >= len(modeltype_names):
        linecolors = plt_colors
    else:
        linecolors = ['b', 'r', 'g', 'y', 'm', 'c', 'k', 'b']

    if use_seaborn:
        linecolors = sns.color_palette("Paired", len(modeltype_names))
    # plt.figure(figsize=(30, 30))
    fig, axs = plt.subplots(2, 2, constrained_layout=True)

    for i in range(len(modeltype_names)):
        for j in range(len(attacks)):
            seed1, seed2, seed3 = accuracies[i]
            acc = np.stack([seed1[j], seed2[j], seed3[j]])

            mean = np.mean(acc, axis=0)

            if normalize:
                mean = mean / np.max(mean)

            std = np.std(acc, axis=0)

            """print()
            print('model: ', modeltype_names[i])
            print('mean: ', np.around(mean, 4))
            print('std: ', np.around(std, 4))"""

            if j == 0:
                pos = (0, 0)
            elif j == 1:
                pos = (0, 1)
            elif j == 2:
                pos = (1, 0)
            elif j == 3:
                pos = (1, 1)

            if len(modeltype_names) > 4:
                axs[pos].plot(epsilons, mean, "*-", color=linecolors[i])
            else:
                axs[pos].plot(epsilons, mean, "*-", linestyle=linetype[i], color=linecolors[i])

            axs[pos].fill_between(epsilons, mean - std, mean + std, alpha=0.5, color=linecolors[i])

            if normalize:
                axs[pos].set_yticks(np.arange(0, 1.1, step=0.1))
            else:
                axs[pos].set_yticks(np.arange(0, 1., step=0.1))

            # plt.title("Accuracy vs Epsilon")
            axs[pos].set_title("Attack: " + attacks[j], fontsize=fontsize)
            axs[pos].set_xlabel("Epsilon", fontsize=fontsize)
            axs[pos].set_ylabel("Accuracy", fontsize=fontsize)
    axs[(0, 1)].legend(labels=modeltype_names, fontsize=fontsize, loc='upper right')
    #axs[(1, 1)].set_axis_off()
    plt.show()


def plot_eps_vs_accuracy_multiple_nosparsefool(epsilons, accuracies, attacks, modeltype_names, normalize=False):
    linetype = ['-', '--', ':', '-.']
    linecolors = ['b', 'r', 'g', 'y', 'm', 'c', 'k', 'b']
    # plt.figure(figsize=(30, 30))
    fig, axs = plt.subplots(3, 1, constrained_layout=True)

    for i in range(len(modeltype_names)):
        for j in range(len(attacks)):
            seed1, seed2, seed3 = accuracies[i]
            acc = np.stack([seed1[j], seed2[j], seed3[j]])

            mean = np.mean(acc, axis=0)

            if normalize:
                mean = mean / np.max(mean)

            std = np.std(acc, axis=0)

            """print()
            print('model: ', modeltype_names[i])
            print('mean: ', np.around(mean, 4))
            print('std: ', np.around(std, 4))"""

            """if j == 0:
                pos = (0, 0)
            elif j == 1:
                pos = (0, 1)
            elif j == 2:
                pos = (1, 0)
            elif j == 3:
                pos = (1, 1)"""

            if len(modeltype_names) > 4:
                axs[j].plot(epsilons, mean, "*-", label=modeltype_names[i], color=linecolors[i])
            else:
                axs[j].plot(epsilons, mean, "*-", label=modeltype_names[i],
                            linestyle=linetype[i], color=linecolors[i])

            axs[j].fill_between(epsilons, mean - std, mean + std, alpha=0.5, color=linecolors[i])

            if normalize:
                axs[j].set_yticks(np.arange(0, 1.1, step=0.1))
            else:
                axs[j].set_yticks(np.arange(0, 1., step=0.1))

            # plt.title("Accuracy vs Epsilon")
            axs[j].set_title("Attack: " + attacks[j])
            axs[j].set_xlabel("Epsilon")
            axs[j].set_ylabel("Accuracy")
            axs[j].legend()
    plt.show()


def make_boxplot(models, attacks, accuracies, epsilon):
    fig, axs = plt.subplots(nrows=len(models), ncols=len(attacks), figsize=(20, 10))
    fig.suptitle('some title')
    for i in range(len(models)):
        for j in range(len(attacks)):
            ax = axs[i, j]
            seed1, seed2, seed3 = accuracies[i]
            acc = np.stack([seed1[j], seed2[j], seed3[j]])
            # rectangular box plot
            bplot1 = ax.boxplot(acc,
                                vert=True,  # vertical box alignment
                                patch_artist=True,  # fill with color
                                labels=epsilon)  # will be used to label x-ticks
            ax.set_title(models[i] + ' - ' + attacks[j])
    plt.tight_layout()
    plt.show()


def plot_examples(examples, epsilons):
    # Plot several adversarial samples at each epsilon
    cnt = 0
    plt.figure(figsize=(8, 10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons), len(examples[0]), cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig, adv, ex = examples[i][j]
            plt.title("{} -> {}".format(orig, adv))
            plt.imshow(ex)
    plt.tight_layout()
    plt.show()


def plot_bird_results(model_names, datasets, accuracies, normalize=False, ylabel='Accuracy'):
    linetype = ['-', '--', ':', '-.']
    linecolors = ['b', 'r', 'g', 'm', 'y', 'c', 'k', 'w']
    fig, ax = plt.subplots()

    for i in range(len(model_names)):
        means = []
        stds = []
        for j in range(len(datasets)):
            seed1, seed2, seed3 = accuracies[i]
            acc = np.stack([seed1[j], seed2[j], seed3[j]])
            means.append(np.mean(acc, axis=0))
            stds.append(np.std(acc, axis=0))

        means = np.asarray(means)
        stds = np.asarray(stds)

        if normalize:
            means = means / np.max(means)

        print(f'{model_names[i]}: {means}')

        #ax.plot(datasets, means, "*-", label=model_names[i], linestyle=linetype[i], color=linecolors[i])
        ax.plot(datasets, means, "*-", label=model_names[i], color=linecolors[i])
        ax.fill_between(datasets, means - stds, means + stds, color=linecolors[i], alpha=0.5)
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.title(f"{ylabel} on different Datasets")
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def plot_birds_acc_and_norm_results(model_names, datasets, accuracies, ylabel='Accuracy'):
    linetype = ['-', '--', ':', '-.']
    linecolors = ['b', 'r', 'g', 'm', 'y', 'c', 'k', 'w']

    fig, axs = plt.subplots(2, 1, constrained_layout=True)

    for i in range(len(model_names)):
        means = []
        stds = []
        for j in range(len(datasets)):
            seed1, seed2, seed3 = accuracies[i]
            acc = np.stack([seed1[j], seed2[j], seed3[j]])
            means.append(np.mean(acc, axis=0))
            stds.append(np.std(acc, axis=0))

        means = np.asarray(means)
        stds = np.asarray(stds)

        print(f'{model_names[i]}: {means}')

        #axs[0].plot(datasets, means, "*-", linestyle=linetype[i], color=linecolors[i])
        axs[0].plot(datasets, means, "*-", color=linecolors[i])
        axs[0].fill_between(datasets, means - stds, means + stds, color=linecolors[i], alpha=0.5)

        means = means / np.max(means)

        axs[1].plot(datasets, means, "*-", linestyle=linetype[i], color=linecolors[i])
        axs[1].fill_between(datasets, means - stds, means + stds, color=linecolors[i], alpha=0.5)

    axs[0].set_yticks(np.arange(0, 1.1, step=0.1))
    axs[0].set_ylabel(ylabel)
    axs[1].set_yticks(np.arange(0, 1.1, step=0.1))
    axs[1].set_ylabel(f"normalized {ylabel}")
    plt.suptitle(f"{ylabel} on different Datasets")

    fig.legend(labels=model_names)
    plt.show()


def plot_birds_result_difference(model_names, datasets, accuracies, model_type=None):
    linetype = ['-', '--', ':', '-.']
    linecolors = ['b', 'r', 'g', 'y', 'm', 'c', 'k', 'w']

    model_accs = []
    for i in range(len(model_names)):
        dataset_accs = []
        for j in range(len(datasets)):
            seed1, seed2, seed3 = accuracies[i]
            acc = np.stack([seed1[j], seed2[j], seed3[j]])
            dataset_accs.append(acc)
        model_accs.append(dataset_accs)

    # the first part only goes for ConceptBottleneck
    if model_type == 'cb':
        plot_cb_birds_difference_from_standard(model_names=model_names, datasets=datasets, model_accs=model_accs)

    # for scouter and cb model!
    # Ansatz über accs_standard - accs_other, dann mean/std berechnen
    diffs_mean = []
    diffs_std = []
    diff_from_cub = []
    for i in range(len(model_names)):
        model_diff_from_cub = []
        for j in range(1, len(datasets)):
            model_diff_from_cub.append(np.stack(model_accs[i][0] - model_accs[i][j]))
        model_diff_from_cub = np.stack(model_diff_from_cub)

        diff_from_cub.append(model_diff_from_cub)
        diffs_mean.append(np.mean(model_diff_from_cub, axis=0))
        diffs_std.append(np.std(model_diff_from_cub, axis=0))

    # plot mean and std
    fig, ax = plt.subplots()
    for i in range(len(diff_from_cub)):
        ax.plot(datasets[1:], diffs_mean[i], "*-", label=model_names[i], linestyle=linetype[i],
                color=linecolors[i])
        ax.fill_between(datasets[1:], diffs_mean[i] - diffs_std[i], diffs_mean[i] + diffs_std[i],
                        color=linecolors[i], alpha=0.5)

    plt.title("Difference in Accuracy from CUB_200_2011 Dataset")
    plt.ylabel("Difference in Accuracy")
    plt.xlabel('Dataset')
    plt.legend()
    plt.show()


def plot_cb_birds_difference_from_standard(model_names, datasets, model_accs):
    linetype = ['-', '--', ':', '-.']
    linecolors = ['b', 'r', 'g', 'y', 'm', 'c', 'k', 'w']

    # Ansatz über accs_standard - accs_other, dann mean/std berechnen
    model_names_other = [m for m in model_names if not ('standard' in m or 'inception' in m)]

    for el in model_names:
        if 'standard' in el or 'inception' in el:
            idx = model_names.index(el)
    stand = np.stack(model_accs[idx], axis=1)

    diffs_from_standard = []
    diffs_mean = []
    diffs_std = []
    for name in model_names_other:
        diff = stand - np.stack(model_accs[model_names.index(name)], axis=1)

        diffs_mean.append(np.mean(diff, axis=0))
        diffs_std.append(np.std(diff, axis=0))
        diffs_from_standard.append(diff)

    # plot mean and std
    fig, ax = plt.subplots()
    for i in range(len(diffs_from_standard)):
        ax.plot(datasets, diffs_mean[i], "*-", label=model_names_other[i], linestyle=linetype[i + 1],
                color=linecolors[i + 1])
        ax.fill_between(datasets, diffs_mean[i] - diffs_std[i], diffs_mean[i] + diffs_std[i], color=linecolors[i + 1],
                        alpha=0.5)

    plt.title("Difference in Accuracy from Standard Model vs Dataset")
    plt.ylabel("Difference in Accuracy")
    plt.xlabel('Dataset')
    plt.legend()
    plt.show()


def plot_travBirds_bird_accuracies_maskrcnn(accuracy_types, datasets, accuracies, normalize=False, fix_yticks=True,
                                            ytitle='Accuracy'):
    linetype = ['-', '--', ':', '-.']
    linecolors = ['b', 'r', 'g', 'm', 'y', 'c', 'k', 'w']
    fig, ax = plt.subplots()

    for i in range(len(accuracy_types)):
        if normalize:
            score = accuracies[i] / np.max(accuracies[i])
        else:
            score = accuracies[i]

        print(f"Model: Mask R-CNN \t Accuracy type: {accuracy_types[i]}\t Attack: {datasets}\t Score: {accuracies[i]}")

        ax.plot(datasets, score, "*-", label=accuracy_types[i], linestyle=linetype[i], color=linecolors[i])

    if fix_yticks:
        plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.title(f"Mask R-CNN: {ytitle} on different datasets")
    plt.ylabel(ytitle)
    plt.xlabel("Datasets")
    plt.legend()
    plt.show()


def plot_travBirds_bird_not_found_maskrcnn(model_names, datasets, accuracies, normalize=False, fix_yticks=True,
                                           ytitle='Accuracy'):
    linetype = ['-', '--', ':', '-.']
    linecolors = ['b', 'r', 'g', 'm', 'y', 'c', 'k', 'w']
    fig, ax = plt.subplots()

    if normalize:
        score = accuracies / np.max(accuracies)
    else:
        score = accuracies

    print(f"Model: {model_names}\t Attack: {datasets}\t Score: {accuracies}")

    ax.plot(datasets, accuracies, "*-", label=model_names, linestyle=linetype[0], color=linecolors[0])

    ax.set_ylim(ymin=0)
    plt.title(f"Mask R-CNN: {ytitle} on different datasets")
    plt.ylabel(ytitle)
    plt.xlabel("Datasets")
    plt.legend()
    plt.show()


def plot_adv_eps_vs_accuracy_maskrcnn(epsilons, accuracies, attacks, modeltype_names, accuracy_types, normalize=False,
                                      use_seaborn=False, fix_yticks=True, ytitle='Accuracy'):
    linetype = ['-', '--', ':', '-.']
    linecolors = ['b', 'r', 'g', 'y', 'm', 'c', 'k', 'b']

    accuracies = np.array(accuracies)

    if use_seaborn:
        linecolors = sns.color_palette("Paired", len(modeltype_names))
    fig, axs = plt.subplots(2, 2, constrained_layout=True)

    for j in range(len(attacks)):
        if j == 0:
            pos = (0, 0)
        elif j == 1:
            pos = (0, 1)
        elif j == 2:
            pos = (1, 0)
        elif j == 3:
            pos = (1, 1)

        for i in range(len(accuracy_types)):
            scores = accuracies[i, j]

            if normalize:
                scores = scores / np.max(scores)

            print(
                f"Model: {modeltype_names}\t Attack: {attacks[j]}\t Accuracy Type: {accuracy_types[i]}\t Score: {scores}")
            axs[pos].plot(epsilons, scores, "*-", label=accuracy_types[i],
                          linestyle=linetype[i], color=linecolors[i])

        if fix_yticks:
            axs[pos].set_yticks(np.arange(0, 1.1, step=0.1))

        axs[pos].set_title("Attack: " + attacks[j])
        axs[pos].set_xlabel("Epsilon")
        axs[pos].set_ylabel(ytitle)
        axs[pos].legend()
    if 'SparseFool' not in attacks:
        axs[1, 1].set_axis_off()
    plt.suptitle(f"Mask R-CNN: {ytitle} vs Epsilon")
    plt.show()


def plot_adv_eps_vs_others_maskrcnn(epsilons, accuracies, attacks, modeltype_names, normalize=False, use_seaborn=False,
                                    ytitle='Accuracy'):
    linetype = ['-', '--', ':', '-.']
    linecolors = ['b', 'r', 'g', 'y', 'm', 'c', 'k', 'b']

    if use_seaborn:
        linecolors = sns.color_palette("Paired", len(modeltype_names))
    fig, axs = plt.subplots(1, 1, constrained_layout=True)

    for j in range(len(attacks)):
        scores = accuracies[j]

        if normalize:
            scores = scores / np.max(scores)

        print(f"Model: {modeltype_names}\t Attack: {attacks[j]}\t Score: {scores}")

        axs.plot(epsilons, scores, "*-", label=attacks[j],
                 linestyle=linetype[j], color=linecolors[j])

        axs.set_xlabel("Epsilon")
        axs.set_ylabel(ytitle)
        axs.legend()
    plt.suptitle(f"Mask R-CNN: {ytitle} vs Epsilon")
    plt.show()


def plot_mask_stage2_eps_vs_accuracy_multiple(epsilons, accuracies, attacks, modeltype_names, normalize=False,
                                              use_seaborn=False, ylabel='Accuracy', fontsize='medium'):
    linetype = ['-', '--', ':', '-.']
    linecolors = ['b', 'r', 'g', 'y', 'm', 'c', 'k', 'b']

    if use_seaborn:
        linecolors = sns.color_palette("Paired", len(modeltype_names))
    fig, axs = plt.subplots(2, 2, constrained_layout=True)

    for i in range(len(modeltype_names)):
        for j in range(len(attacks)):
            seed1, seed2, seed3 = accuracies[i]
            acc = np.stack([seed1[j], seed2[j], seed3[j]])
            mean = np.mean(acc, axis=0)
            if normalize:
                mean = mean / np.max(mean)
            std = np.std(acc, axis=0)

            print(
                f"Model: {modeltype_names[i]}\t Attack: {attacks[j]}\t Accuracy Type: {ylabel}\t Score: {mean}")

            if j == 0:
                pos = (0, 0)
            elif j == 1:
                pos = (0, 1)
            elif j == 2:
                pos = (1, 0)
            elif j == 3:
                pos = (1, 1)

            if len(modeltype_names) > 4:
                axs[pos].plot(epsilons, mean, "*-", color=linecolors[i])
            else:
                axs[pos].plot(epsilons, mean, "*-", linestyle=linetype[i], color=linecolors[i])
            axs[pos].fill_between(epsilons, mean - std, mean + std, alpha=0.5, color=linecolors[i])

            if normalize:
                axs[pos].set_yticks(np.arange(0, 1.1, step=0.1))
            else:
                axs[pos].set_yticks(np.arange(0, 1., step=0.1))
            axs[pos].set_title("Attack: " + attacks[j])
            axs[pos].set_xlabel("Epsilon")
            axs[pos].set_ylabel(ylabel)
    fig.legend(labels=modeltype_names, fontsize=fontsize, loc='lower right')
    plt.suptitle(f"{ylabel} vs Epsilon")
    if 'SparseFool' not in attacks:
        axs[1, 1].set_axis_off()
    plt.show()