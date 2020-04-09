import os

import numpy as np
from real_eigen import GATE_FACTOR
import matplotlib.pyplot as plt


# hyper parameters, model, history data, filename

plt.style.use("ggplot")

def plt_heatmap(ax, mat):
    pass

def np2latex(m):

  # return "\\hline\n" + " \\\\\n\\hline\n".join([" & ".join(map(str,line)) for line in m]) + " \\\\\n\\hline"
  return "\\\\\n".join([" & ".join(map(str,[c for c in line])) for line in m])

def plot_images(x_plots, y_plots, images, history_root, image_tag, hspace=None, wspace=None, figsize=None, dpi=None):
    save_path = os.path.join(history_root, f"{image_tag}.eps")

    fig = plt.figure()
    # plt.autoscale(True)
    if figsize != None:
        fig.set_size_inches(*figsize, forward=True)
    plt.axis("off")

    for i in range(x_plots*y_plots):

        ax = fig.add_subplot(x_plots, y_plots, i+1)
        # ax.autoscale(True)
        ax.axis("off")

        ax.imshow(images[i], cmap="gray")

    if hspace != None:
        plt.subplots_adjust(hspace=hspace)
    if wspace != None:
        plt.subplots_adjust(wspace=wspace)
    if dpi == None:
        plt.savefig(save_path)
    else:
        plt.savefig(save_path, dpi=dpi)
    plt.close()

def plot_images_with_title(x_plots, y_plots, images, history_root, image_tag, hspace=None, wspace=None):
    save_path = os.path.join(history_root, f"{image_tag}.eps")

    fig, axes = plt.subplots(x_plots, y_plots, sharey="row")
    fig.set_size_inches(20, 20, forward=True)
    # plt.autoscale(True)
    # plt.axis("off")

    # for i in range(x_plots*y_plots):

    #     ax = fig.add_subplot(x_plots, y_plots, i+1)
    #     # ax.autoscale(True)
    #     ax.axis("off")

    #     ax.imshow(images[i], cmap="gray")

    for i in range(x_plots):
        for j in range(y_plots):
            axes[i, j].axis("off")
            axes[i, j].imshow(images[i*y_plots+j], cmap="gray")
        axes[i, 0].set_axis_on()
        axes[i, 0].set_ylabel(f"class {i}", fontsize=24, rotation=90)
        axes[i, 0].grid(False)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

    if hspace != None:
        plt.subplots_adjust(hspace=hspace)
    if wspace != None:
        plt.subplots_adjust(wspace=wspace)
    plt.savefig(save_path, dpi=200)
    plt.close()


def get_ecmm_nonoverlap(LL):
    LL_overlap_num = np.sum(LL, axis=1)
    LL_nonoverlap = LL.copy()
    LL_nonoverlap[LL_overlap_num > 1, :]  = LL_nonoverlap[LL_overlap_num > 1, :] * 0
    return LL_overlap_num, LL_nonoverlap

def get_to_plot_change_of_basis(x_train, y, P, num_classes, with_original=True):

    x_train_change_of_basis = x_train.dot(P)

    to_plot_cob_with_original = np.zeros(x_train_change_of_basis[:100, :, :].shape, dtype=x_train.dtype)
    to_plot_cob = np.zeros(x_train_change_of_basis[:100, :, :].shape, dtype=x_train.dtype)
    for i in range(num_classes):
        to_plot_cob_with_original[i*10:(i+1)*10-5, :, :] = x_train_change_of_basis[y==i, :, :][:5, :, :]
        to_plot_cob_with_original[(i+1)*10-5:(i+1)*10, :, :] = x_train[y==i, :, :][:5, :, :]

        to_plot_cob[i*10:(i+1)*10, :, :] = x_train_change_of_basis[y == i, :, :][:10, :, :]


    return to_plot_cob, to_plot_cob_with_original, x_train_change_of_basis

def get_merge_images(x_cob, y, P, LL_mask, weights_type="mean"):
    # distribution of projection on each eigenfeature of each class, 
    # use the mean of that distribution to sum eigenfeature; 
    # pure eigenfeature and overlapped

    state_len = P.shape[0]
    side_len = int(np.sqrt(state_len))
    num_classes = LL_mask.shape[1]

    all_weights = np.zeros((num_classes, state_len), dtype=x_cob.dtype)
    if weights_type == "mean":
        for i in range(num_classes):
            cob_of_class = x_cob[y == i]
            all_weights[i] = np.mean(cob_of_class, axis=0) 

    else:
        for i in range(num_classes):
            cob_of_class = x_cob[y == i]
            for j in range(28*28):
                hist, bin_edges = np.histogram(cob_of_class[:, 0, j], bins="auto")
                all_weights[i, j] = bin_edges[np.argmax(hist)]


    all_weights = all_weights * LL_mask.T
    all_images = all_weights.dot(P.T)

    return all_images

def plot_cob(to_plot_cob, dim_x, dim_y, num_classes, history_root):
    plot_images_with_title(num_classes, num_classes, to_plot_cob.reshape(-1, dim_x, dim_y), 
        history_root, "digit_change_of_basis", hspace=0.1, wspace=0.1)
    
def plot_cob_with_original(to_plot_cob, dim_x, dim_y, num_classes, history_root):
    plot_images_with_title(num_classes, num_classes, to_plot_cob.reshape(-1, dim_x, dim_y), 
        history_root, "digit_change_of_basis_with_original", hspace=0.1, wspace=0.1)

def plot_weighted_sum(x_cob, y, P, LL, LL_nonoverlap, dim_x, dim_y, history_root):


    all_images_mean_overlap = get_merge_images(x_cob, y, P, LL, weights_type="mean")
    plot_images(2, 5, all_images_mean_overlap.reshape(-1, dim_x, dim_y), 
        history_root, "mean_weighted_sum_of_all_eigenfeature", hspace=0.6, wspace=0.1, figsize=(10,3), dpi=400)

    all_images_mean_pure = get_merge_images(x_cob, y, P, LL_nonoverlap, weights_type="mean")
    plot_images(2, 5, all_images_mean_pure.reshape(-1, dim_x, dim_y), 
        history_root, "mean_weighted_sum_of_pure_eigenfeature", hspace=0.6, wspace=0.1, figsize=(10,3), dpi=400)

    all_images_peak_overlap = get_merge_images(x_cob, y, P, LL, weights_type="peak")
    plot_images(2, 5, all_images_peak_overlap.reshape(-1, dim_x, dim_y), 
        history_root, "peak_weighted_sum_of_all_eigenfeature", hspace=0.6, wspace=0.1, figsize=(10,3), dpi=400)
    all_images_peak_pure = get_merge_images(x_cob, y, P, LL_nonoverlap, weights_type="peak")
    plot_images(2, 5, all_images_peak_pure.reshape(-1, dim_x, dim_y), 
        history_root, "peak_weighted_sum_of_pure_eigenfeature", hspace=0.6, wspace=0.1, figsize=(10,3), dpi=400)

def get_proj_dist(x_train_change_of_basis, LL_overlap_num, LL_nonoverlap):
    #global pure_digit

    # projections on one eigenfeature (least overlapped, most overlapped, 1 and 8, 4 plots) each class with different color

    proj_pure_eigenfeature_all = x_train_change_of_basis[:, :, LL_overlap_num == 1]
    proj_most_overlap_eigenfeature_all = x_train_change_of_basis[:, :, LL_overlap_num >= 5]


    # pure_digit = [0, 1, 2, 5, 7, 8]
    pure_digit = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    pure_ef_idx = []

    for i, v in enumerate(LL_nonoverlap):
        # if i == 2 or i == 5:
        #     continue
        digit = np.argmax(v)
        if digit == 0 and v[0] == 0:
            continue

        if digit in pure_digit:
            pure_ef_idx.append((digit, i))
            pure_digit.pop(pure_digit.index(digit))    

    print(pure_ef_idx)

    pure_ef_idx = list(zip(*sorted(pure_ef_idx)))[1]
    pure_projs = x_train_change_of_basis[:, 0, pure_ef_idx]
    print(pure_projs.shape)
    two_overlap_projs = x_train_change_of_basis[:, :, LL_overlap_num==2][:, 0, 9:10]
    print(two_overlap_projs.shape)
    three_overlap_projs = x_train_change_of_basis[:, :, LL_overlap_num==3][:, 0, 9:10]
    print(three_overlap_projs.shape)
    four_overlap_projs = x_train_change_of_basis[:, :, LL_overlap_num==4][:, 0, 9:10]
    print(four_overlap_projs.shape)
    most_overlap_projs = proj_most_overlap_eigenfeature_all[:, 0, 0:3]
    print(most_overlap_projs.shape)
    to_plot_proj_dist = np.concatenate([
        pure_projs, two_overlap_projs, 
        three_overlap_projs, four_overlap_projs, 
        most_overlap_projs], axis=1)

    print(to_plot_proj_dist.shape)

    return to_plot_proj_dist

def plot_proj_dist(all_projs, y, num_classes, x_plots, y_plots, history_root, density=False):
    freq_or_density = ("freq", "density")[density]
    proj_dist_save_path = os.path.join(history_root, f"proj_{freq_or_density}_dist_ef.png")

    # fig = plt.figure(figsize=(10, 7))
    fig, axes = plt.subplots(x_plots,y_plots)
    fig.tight_layout()
    fig.set_size_inches(13, 13, forward=True)
    # plt.autoscale(True)

    pure_digit = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    titles = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]
    for i in range(x_plots):
        for j in range(y_plots):
            idx = i*3+j
            class_label = pure_digit[idx]
            axes[i, j].set_xlabel(f"Projection on a PE (class '{class_label})'")

            # if i < 2:
            #     axes[i, j].set_xlabel("Projection on a PE")
            # elif i == 2:
            #     axes[i, j].set_xlabel(f"Projection on a ${2+j}^o$ eigenfeature")
            # else:
            #     if j < 2:
            #         axes[i, j].set_xlabel("Projection on a $5^o$ eigenfeature")
            #     else:
            #         axes[i, j].set_xlabel("Projection on a $6^o$ eigenfeature")


            if density:
                axes[i, j].set_ylabel("Density")
            else:
                axes[i, j].set_ylabel("Frequency")

            axes[i, j].set_title(f"({titles[idx]})")

    idx = 0
    for i in range(x_plots):
        for j in range(y_plots):
            axes[i, j].set_xlabel(f"Projection on eigenfeature {i}")
            axes[i, j].set_ylabel("Frequency")

            for k in range(num_classes):
                # print(idx)
                axes[i, j].hist(all_projs[:, idx][y == k], bins="auto", alpha=0.1*(10-k/2), density=density)
            leg = axes[i, j].legend([f"class {l}" for l in range(num_classes)], 
                fancybox=True, framealpha=0.5, loc="best")

            # for k, lh in enumerate(leg.legendHandles):
            #     lh.set_alpha(1-k*0.05)

            idx += 1

    plt.subplots_adjust(hspace=0.4, wspace=0.2)
    plt.savefig(proj_dist_save_path, dpi=300)
    plt.close()




def plot_proj_dist_yinheng(dataset, num_classes, P, LL, history_root, density=False):
    freq_or_density = ("freq", "density")[density]
    proj_dist_save_path = os.path.join(history_root, f"proj_{freq_or_density}_dist_ef.png")

    x_train, y_train, x_test, y_test = dataset
    x_train = np.squeeze(x_train)
    x_test = np.squeeze(x_test)
    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)

    x_train /= np.linalg.norm(x_train, axis=1, keepdims=True)
    x_test /= np.linalg.norm(x_test, axis=1, keepdims=True)

    x_train_projs = x_train.dot(P)
    x_test_projs = x_test.dot(P)
    x_projs = np.concatenate([x_train, x_test], axis=0).dot(P)

    print(x_train_projs.shape)

    y = np.concatenate([y_train, y_test], axis=0)

    x_plots = P.shape[0]
    y_plots = 3

    fig, axes = plt.subplots(x_plots, y_plots)
    fig.tight_layout()
    fig.set_size_inches(13, 13, forward=True)

    for i in range(x_plots):
        for j in range(y_plots):
            for k in range(num_classes):
                if j == 0:
                    axes[i, j].hist(x_train_projs[:, i][y_train == k], bins="auto", alpha=0.5, density=density)
                elif j == 1:
                    axes[i, j].hist(x_test_projs[:, i][y_test == k], bins="auto", alpha=0.5, density=density)
                else:
                    axes[i, j].hist(x_projs[:, i][y == k], bins="auto", alpha=0.5, density=density)

            leg = axes[i, j].legend([f"class {l}" for l in range(num_classes)], 
                fancybox=True, framealpha=0.5, loc="best")



    plt.subplots_adjust(hspace=0.4, wspace=0.2)
    plt.savefig(proj_dist_save_path, dpi=300)
    plt.close()






# def plot_prob_dist(all_probs, y, num_classes, x_plots, y_plots, history_root, density=False):
#     freq_or_density = ("freq", "density")[density]
#     proj_dist_save_path = os.path.join(history_root, f"prob_{freq_or_density}_dist_ef.png")

#     # fig = plt.figure(figsize=(10, 7))
#     fig, axes = plt.subplots(x_plots,y_plots)
#     fig.tight_layout()
#     fig.set_size_inches(13, 13, forward=True)
#     # plt.autoscale(True)

#     titles = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]
#     for i in range(x_plots):
#         for j in range(y_plots):
#             if i < 2:
#                 axes[i, j].set_xlabel("Probability on a PE")
#             elif i == 2:
#                 axes[i, j].set_xlabel(f"Probability on a ${2+j}^o$ eigenfeature")
#             else:
#                 if j < 2:
#                     axes[i, j].set_xlabel("Probability on a $5^o$ eigenfeature")
#                 else:
#                     axes[i, j].set_xlabel("Probability on a $6^o$ eigenfeature")


#             if density:
#                 axes[i, j].set_ylabel("Density")
#             else:
#                 axes[i, j].set_ylabel("Frequency")

#             axes[i, j].set_title(f"({titles[i*3+j]})")

#     idx = 0
#     for i in range(x_plots):
#         for j in range(y_plots):
#             for k in range(num_classes):
#                 # print(idx)
#                 axes[i, j].hist(all_probs[:, idx][y == k], bins="auto", alpha=0.1*(10-k), density=density)
#             leg = axes[i, j].legend([f"class {l}" for l in range(num_classes)], 
#                 fancybox=True, framealpha=0.5, loc="best")

#             # for k, lh in enumerate(leg.legendHandles):
#             #     lh.set_alpha(1-k*0.05)

#             idx += 1

#     plt.subplots_adjust(hspace=0.4, wspace=0.2)
#     plt.savefig(prob_dist_save_path, dpi=300)
#     plt.close()



def save_eigenfeature(P, LL, history_root):

    state_len = P.shape[0]
    dim = np.sqrt(state_len).astype(int)

    num_plots = 10
    all_plots = num_plots * num_plots

    for i in range(state_len//all_plots):

        to_plot_all = P.T[i*all_plots:(i+1)*all_plots, :] 

        plot_images(num_plots, num_plots, to_plot_all.reshape(-1, dim, dim), 
            history_root, f"eigenfeature-{i}.eps", hspace=0.1, wspace=0.1, figsize=(10, 10), dpi=400)


def save_nonoverlap(P, LL, history_root):
    LL_overlap_num = np.sum(LL, axis=1)
    P = P[:, LL_overlap_num == 1]

    num_nonoverlap_ef = P.shape[1]
    state_len = P.shape[0]
    dim = np.sqrt(state_len).astype(int)

    num_plots = 10
    all_plots = num_plots*num_plots

    for i in range(num_nonoverlap_ef//all_plots+1):

        to_plot_nonoverlap = P.T[i*all_plots:(i+1)*all_plots, :] 
        
        if to_plot_nonoverlap.shape[0] < all_plots:
            break

        plot_images(num_plots, num_plots, to_plot_nonoverlap.reshape(-1, dim, dim), 
            history_root, f"nonoverlap-{i}.eps", hspace=0.1, wspace=0.1, figsize=(10, 10), dpi=400)


# def save_merged(P, LL, history_root):
#   LL2 = ["".join([str(int(b)) for b in row]) for row in LL]
#   LL2 = [int(d[::-1], 2) for d in LL2]
#   LL2 = np.array(LL2)

#   dim = np.sqrt(LL.shape[0])
#   dim = dim.astype(int)

#   history_save_path = os.path.join(history_root, f"eigenmerged.eps")

#   fig = plt.figure()
#   # plt.autoscale(True)
#   plt.axis("off")
#   # plt.figure(figsize=(6.4,6.4))
#   for i in range(10):
#     pc = P[:, LL2 == np.math.pow(2,i)]
#     print(pc.shape)

#     ax = fig.add_subplot(2, 5, i+1)
#     # ax.autoscale(True)
#     ax.axis("off")

#     pc = np.sum(pc, axis=1)
#     img = pc.reshape(dim, dim)
#     # img_min = np.min(img)
#     # img_max = np.max(img)
#     # img -= img_min
#     # img *= 255/img_max
#     ax.imshow(img, cmap="gray")
#     #ax.imshow(img)

#   plt.subplots_adjust(hspace=0, wspace=0.2)
#   plt.savefig(history_save_path)
#   plt.close()

# def save_merged_all(P, LL, history_root):
#   LL = LL.astype(int)

#   dim = np.sqrt(LL.shape[0])
#   dim = dim.astype(int)

#   history_save_path = os.path.join(history_root, f"eigenmergedall.eps")

#   fig = plt.figure()
#   # plt.autoscale(True)
#   plt.axis("off")
#   # plt.figure(figsize=(6.4,6.4))
#   for i in range(10):
#     pc = P[:, LL[:,i]==1]
#     print(pc.shape)

#     ax = fig.add_subplot(2, 5, i+1)
#     # ax.autoscale(True)
#     ax.axis("off")

#     pc = np.sum(pc, axis=1)
#     img = pc.reshape(dim, dim)
#     # img_min = np.min(img)
#     # img_max = np.max(img)
#     # img -= img_min
#     # img *= 255/img_max
#     ax.imshow(img, cmap="gray")
#     #ax.imshow(img)

#   plt.subplots_adjust(hspace=0, wspace=0.2)
#   plt.savefig(history_save_path)
#   plt.close()


def save_overlapping(y, history_root):
    y_freq = np.bincount(y.astype(int))
    x = np.arange(np.max(y)+1)

    history_save_path = os.path.join(history_root, "overlapping.eps")
    fig= plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel("Degree of Overlapping")
    ax.set_ylabel("Number of Eigenfeatures")

    #ax.hist(y, align="mid")
    ax.bar(x, height=y_freq)
    #plt.xlabel = "Number of Eigenfeatures"
    #plt.ylabel = "Number of Overlappings"
    plt.savefig(history_save_path)
    plt.close()

def save_crowdedness(num_classes, history_root, weights=[]):
    history_save_path = os.path.join(history_root, "crowdedness.eps")

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel("Class label")
    ax.set_ylabel("Crowdedness (class degeneracy)")

    #ax.hist(x, list(range(-1,bin+1,1)), weights=weights, rwidth=0.8, align="mid")
    ax.bar(list(range(num_classes)), height=weights)
    #plt.xlabel = "Number of Eigenfeatures"
    #plt.ylabel = "Number of Overlappings"
    plt.xticks(list(range(num_classes)))
    plt.savefig(history_save_path)
    plt.close()

def save_degeneracy_eigenvalues(LL, history_root):
    width = len(LL[0])
    
    LL2 = ["".join([str(int(b)) for b in row]) for row in LL]
    # LL2_str = LL2
    LL2 = [int(d[::-1], 2) for d in LL2]
    LL2 = np.array(LL2)


    # history_ll2_save_path = os.path.join(history_root, "ll2.txt")
    history_ll2_freq_save_path = os.path.join(history_root, "ll2_freq.txt")
    history_save_path_eps = os.path.join(history_root, "degeneracy_eigenvalues.eps")

    eigenvalues_freq = np.bincount(LL2)
    eigenvalues = np.arange(np.max(LL2) + 1)
    eigenvalues = eigenvalues[eigenvalues_freq != 0]
    eigenvalues_freq = eigenvalues_freq[eigenvalues_freq != 0]
    eigenvalues_bin_str = ["{0:0{width}b}".format(i, width=width) for i in eigenvalues]
    num_ev = len(eigenvalues)
    ev_pos = np.arange(num_ev)

    ev_freq_pair = zip(eigenvalues, eigenvalues_freq)
    ev_freq_pair_str = []
    for i, row in enumerate(ev_freq_pair):
      row_lst = list(row)
      row_lst.insert(1, eigenvalues_bin_str[i])
      ev_freq_pair_str.append(row_lst)

    ev_freq_pair_str = "\n".join(["\t".join([str(col) for col in row]) for row in ev_freq_pair_str])

    with open(history_ll2_freq_save_path, "w") as fd:
      fd.write(ev_freq_pair_str)

    fig = plt.figure()
    fig.tight_layout()
    ax = fig.add_subplot(111)

    # ax.set_xlabel(r"Eigenvalues of $H_{MNIST}$ ($0 \sim 2^{10} -1 $)")
    ax.set_xlabel("Eigenvalues ($0 \\sim 2^{{{}}} -1 $) of $H$".format(width))
    ax.set_ylabel("Degeneracy")

    #ax.hist(x, list(range(-1,bin+1,1)), weights=weights, rwidth=0.8, align="mid")
    ax.bar(ev_pos, height=eigenvalues_freq)
    ax.set_yscale("log")
    ax.set_xticks(ev_pos)
    ax.set_xticklabels(eigenvalues_bin_str, rotation=-80, fontsize=12)


    plt.subplots_adjust(bottom=0.35)
    plt.savefig(history_save_path_eps, format="eps")
    plt.close()

def save_degeneracy_pure_eigenvalues(LL, num_classes, history_root):
    crowdedness = np.sum(LL, axis=0)

    LL_overlap_num = np.sum(LL, axis=1)
    LL[LL_overlap_num > 1] = LL[LL_overlap_num > 1] * 0

    width = len(LL[0])
    
    LL2 = ["".join([str(int(b)) for b in row]) for row in LL]
    # LL2_str = LL2
    LL2 = [int(d[::-1], 2) for d in LL2]
    LL2 = np.array(LL2)


    # history_ll2_save_path = os.path.join(history_root, "ll2.txt")
    history_ll2_freq_save_path = os.path.join(history_root, "ll2_pure_freq.txt")
    history_save_path_eps = os.path.join(history_root, "degeneracy_and_crowdedness.eps")

    eigenvalues_freq = np.bincount(LL2)
    eigenvalues = np.arange(np.max(LL2) + 1)
    eigenvalues = eigenvalues[eigenvalues_freq != 0]
    eigenvalues_freq = eigenvalues_freq[eigenvalues_freq != 0]
    eigenvalues_bin_str = ["{0:0{width}b}".format(i, width=width) for i in eigenvalues]
    num_ev = len(eigenvalues)
    ev_pos = np.arange(num_ev)

    ev_freq_pair = zip(eigenvalues, eigenvalues_freq)
    ev_freq_pair_str = []
    for i, row in enumerate(ev_freq_pair):
      row_lst = list(row)
      row_lst.insert(1, eigenvalues_bin_str[i])
      ev_freq_pair_str.append(row_lst)

    ev_freq_pair_str = "\n".join(["\t".join([str(col) for col in row]) for row in ev_freq_pair_str])

    with open(history_ll2_freq_save_path, "w") as fd:
      fd.write(ev_freq_pair_str)

    fig = plt.figure()
    fig.tight_layout()
    ax = fig.add_subplot(121)
    ax.set_title("(a)")

    ax.set_xlabel("Eigenvalues of pure eigenfeatures (PEs) of $H$".format(width))
    ax.set_ylabel("Degeneracy")

    ax.bar(ev_pos[1:], height=eigenvalues_freq[1:])
    ax.set_xticks(ev_pos[1:])
    ax.set_xticklabels(eigenvalues_bin_str[1:], rotation=-80, fontsize=12)


    ax2 = fig.add_subplot(122)

    ax2.set_xlabel("Class label")
    ax2.set_ylabel("Crowdedness (class degeneracy)")
    ax2.set_title("(b)")

    ax2.bar(np.arange(num_classes), height=crowdedness)
    ax2.set_xticks(np.arange(num_classes))
    ax2.set_xticklabels(np.arange(num_classes))

    plt.subplots_adjust(bottom=0.3, wspace=0.3)
    plt.savefig(history_save_path_eps, format="eps")
    plt.close()




def save_history(dataset, model, num_classes, history, data_tag, work_magic_code, magic_code, history_root, time_stamp):
    P_save_path = os.path.join(history_root, "P.npy")
    PP_save_path = os.path.join(history_root, "PP.npy")
    L_save_path = os.path.join(history_root, "L.npy")
    LL_save_path = os.path.join(history_root, "LL.npy")

    history_save_path = os.path.join(history_root, "history.npy")
    if history != []:
        np.save(history_save_path, history.history)

    P = model.layers[0].get_weights()[0]
    PP = P.T.dot(P)
    print(P)
    print(PP)
    # print(np2latex(P))

    L = model.layers[2].get_weights()[0]
    LL = 1/(1+np.exp(np.sin(L) * -GATE_FACTOR))
    print(L)
    print(LL)
    # print(np2latex(LL))

    np.save(P_save_path, P)
    np.save(PP_save_path, PP)
    np.save(L_save_path, L)
    np.save(LL_save_path, LL)

    # if data_tag == "mnist":
        # dataset_projection(dataset, num_classes, data_tag, P, np.round(LL), history_root, density=True)
        # dataset_projection(dataset, num_classes, data_tag, P, np.round(LL), history_root, density=False)
        # dataset_probability(dataset, num_classes, data_tag, P, np.round(LL), history_root, density=True)
        # dataset_probability(dataset, num_classes, data_tag, P, np.round(LL), history_root, density=False)

    mapped_num = np.sum(LL)
    print(mapped_num)

    # LL_rounded = np.round(LL)

    overlapping = np.sum(np.round(LL), axis=1)
    save_overlapping(overlapping, history_root)
    print(overlapping)

    degeneracy = np.sum(np.round(LL), axis=0)
    save_crowdedness(num_classes, history_root, weights=degeneracy)

    save_degeneracy_eigenvalues(np.round(LL), history_root)
    save_degeneracy_pure_eigenvalues(np.round(LL), num_classes, history_root)

    if data_tag == "mnist" or data_tag == "cifar":
        
        LL_overlap_num, LL_nonoverlap = get_ecmm_nonoverlap(np.round(LL))

        x_train, y_train, _, _ = dataset
        x_train /= np.linalg.norm(x_train, axis=2, keepdims=True)
        y_train_cat = np.argmax(y_train, axis=1)

        num_classes = LL.shape[-1]
        dim = int(np.sqrt(x_train.shape[-1]))

        to_plot_cob, to_plot_cob_with_original, x_cob = get_to_plot_change_of_basis(x_train, y_train_cat, P.copy(), num_classes, with_original=True)

        plot_cob_with_original(to_plot_cob_with_original, dim, dim, num_classes, history_root)
        plot_cob(to_plot_cob, dim, dim, num_classes, history_root)
        plot_weighted_sum(x_cob, y_train_cat, P, np.round(LL), LL_nonoverlap, dim, dim, history_root)

        print(np.max(LL_overlap_num))
        to_plot_proj_dist = get_proj_dist(x_cob, LL_overlap_num, LL_nonoverlap)
        plot_proj_dist(to_plot_proj_dist, y_train_cat, num_classes, 4, 3, history_root, density=True)
        plot_proj_dist(to_plot_proj_dist, y_train_cat, num_classes, 4, 3, history_root, density=False)

        # save_eigenfeature(P.copy(), np.round(LL), history_root)
        save_nonoverlap(P.copy(), np.round(LL), history_root)
        
        # save_merged(P, LL_rounded, history_root)
        # save_merged_all(P, LL_rounded, history_root)

    if data_tag.startswith("yinheng"):
        plot_proj_dist_yinheng(dataset, num_classes, P, np.round(LL), history_root)





def save_model(model, data_tag, work_magic_code, magic_code):
    pass
