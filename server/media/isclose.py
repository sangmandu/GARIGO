def isClose(loc1, loc2):
    if loc1[0] >= loc2[2]:
      return False
    elif loc1[2] <= loc2[0]:
      return False
    elif loc1[1] <= loc2[3]:
      return False
    elif loc1[3] >= loc2[1]:
      return False
    else:
      areaFunc = lambda top, right, bot, left : (bot-top) * (right-left)
      H = []
      V = []
      H.append(loc1[1])
      H.append(loc1[3])
      H.append(loc2[1])
      H.append(loc2[3])
      V.append(loc1[0])
      V.append(loc1[2])
      V.append(loc2[0])
      V.append(loc2[2])
      H.sort()
      V.sort()
      area = (H[2] - H[1]) * (V[2] - V[1])
      # print("ratio:", area / areaFunc(*loc1) , area / areaFunc(*loc2))
      if area / areaFunc(*loc1) >= 0.4 or area / areaFunc(*loc2) >= 0.4:
        return True
      else:
        return False