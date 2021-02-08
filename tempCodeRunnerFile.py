
    img_bgr = cv2.imread('img/jinho1.jpg')
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    rects, shapes, _ = find_faces(img_rgb)
    descrtipors = encode_faces(img_rgb, shapes)

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img_rgb)

    for i, desc in enumerate(descrtipors):

        found = False

        for name, saved_desc in descs.items():
            # a, b 벡터 사이의 유클리드 거리 구함
            dist = np.linalg.norm([desc] - saved_desc, axis=1)

            if dist < 0.6:
                found = True

                text = ax.text(rects[i][0][0], rects[i][0][1], name,
                               color='b', fontsize=40, fontweight='bold')
                text.set_path_effects([path_effects.Stroke(
                    linewidth=10, foreground='white'), path_effects.Normal()])
                rect = patches.Rectangle(rects[i][0],
                                         rects[i][1][1] - rects[i][0][1],
                                         rects[i][1][0] - rects[i][0][0],
                                         linewidth=2, edgecolor='w', facecolor='none')
                ax.add_patch(rect)

                break

        if not found:
            ax.text(rects[i][0][0], rects[i][0][1], 'unknown',
                    color='r', fontsize=20, fontweight='bold')
            rect = patches.Rectangle(rects[i][0],
                                     rects[i][1][1] - rects[i][0][1],
                                     rects[i][1][0] - rects[i][0][0],
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

    plt.axis('off')
    plt.savefig('result/output.png')
    plt.show()
