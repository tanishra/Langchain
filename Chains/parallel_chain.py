from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text \n {text}.",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template="Generate 5 short questions answers from the following text. \n {text}.",
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document. \n {notes} and {quiz}.",
    input_variables=['notes','quiz']
)

parser = StrOutputParser()

model1 = ChatOpenAI()

model2 = ChatAnthropic(model_name="claude-3-sonnet-20250219")

parallel_chain = RunnableParallel({
    'notes' : prompt1 | model1 | parser,
    "quiz" :  prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """
Event Horizon
This is what makes a black hole black. We can think of the event horizon as the black hole's surface. Inside this boundary, the velocity needed to escape the black hole exceeds the speed of light, which is as fast as anything can go. So whatever passes into the event horizon is doomed to stay inside it - even light. Because light can't escape, black holes themselves neither emit nor reflect it, and nothing about what happens within them can reach an outside observer. But astronomers can observe black holes thanks to light emitted by surrounding matter that hasn't yet dipped into the event horizon.

Accretion Disk
The main light source from a black hole is a structure called an accretion disk. Black holes grow by consuming matter, a process scientists call accretion, and by merging with other black holes. A stellar-mass black hole paired with a star may pull gas from it, and a supermassive black hole does the same from stars that stray too close. The gas settles into a hot, bright, rapidly spinning disk. Matter gradually works its way from the outer part of the disk to its inner edge, where it falls into the event horizon. Isolated black holes that have consumed the matter surrounding them do not possess an accretion disk and can be very difficult to find and study.

If we could see it up close, we'd find that the accretion disk has a funny shape when viewed from most angles. This is because the black hole's gravitational field warps space-time, the fabric of the universe, and light must follow this distorted path. Astronomers call this process gravitational lensing. Light coming to us from the top of the disk behind the black hole appears to form into a hump above it. Light from beneath the far side of the disk takes a different path, creating another hump below. The humps' sizes and shapes change as we view them from different angles, and we see no humps at all when seeing the disk exactly face on.

Event Horizon Shadow
The event horizon captures any light passing through it, and the distorted space-time around it causes light to be redirected through gravitational lensing. These two effects produce a dark zone that astronomers refer to as the event horizon shadow, which is roughly twice as big as the black hole's actual surface.

Photon Sphere
From every viewing angle, thin rings of light appear at the edge of the black hole shadow. These rings are really multiple, highly distorted images of the accretion disk. Here, light from the disk actually orbits the black hole multiple times before escaping to us. Rings closer to the black hole become thinner and fainter.

Doppler Beaming
Viewed from most angles, one side of the accretion disk appears brighter than the other. Near the black hole, the disk spins so fast that an effect of Einstein's theory of relativity becomes apparent. Light streaming from the part of the disk spinning toward us becomes brighter and bluer, while light from the side the disk rotating away from us becomes dimmer and redder. This is the optical equivalent of an everyday acoustic phenomenon, where the pitch and volume of a sound - such as a siren - rise and fall as the source approaches and passes by. The black hole's particle jets show off this effect even more dramatically.
"""
result = chain.invoke({'text' : text})

print(result)

chain.get_graph().print_ascii()